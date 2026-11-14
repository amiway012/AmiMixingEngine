//! Time stretching and pitch shifting (spec section 4).
//!
//! * `wsola_stretch` — Waveform Similarity Overlap-Add: keeps pitch, changes
//!   duration. OLA with sin^2 windows and a similarity search for the optimal
//!   overlap offset gives clean results on percussive & tonal material within
//!   the recommended ±8% range.
//! * `pitch_shift_vocode` — phase-vocoder style: resynthesizes via STFT
//!   magnitude/phase propagation + WSOLA-free scale-and-recompress; simple but
//!   adequate for ±1–2 semitone key shifts used in harmonic mixing.

use crate::dsp::overlapping_cos_window;

/// Stretch `input` by `ratio` (>1 = longer) preserving pitch.
/// `frame_size` ~ 50 ms is a good default for music.
pub fn wsola_stretch(input: &[f32], ratio: f64, sample_rate: f64) -> Vec<f32> {
    if ratio.abs() < 1e-9 || (ratio - 1.0).abs() < 1e-6 || input.len() < 4 {
        return input.to_vec();
    }
    let frame = ((sample_rate * 0.05) as usize).next_power_of_two().max(256); // ~50-70ms
    let hop_out = (frame as f64 / 2.0 / ratio) as usize; // output hop follows ratio
    let hop_out = hop_out.max(1);
    let win = overlapping_cos_window(frame);
    let max_shift = (frame as f64 * 0.25) as usize; // search window for best overlap
    let mut out = vec![0.0f32; (input.len() as f64 * ratio).ceil() as usize + frame];
    let mut norm = vec![0.0f32; out.len()];
    let mut read = 0usize; // position of current analysis frame start
    let mut write = 0usize;
    let mut prev_tail: Vec<f32> = Vec::new();
    while read + frame < input.len() {
        // find shift d maximizing cross-correlation between previous tail and
        // candidate region starting at read + hop_in - frame/2 + d
        let mut best_d = 0isize;
        let mut best_score = f64::NEG_INFINITY;
        if !prev_tail.is_empty() {
            let base = read as isize - prev_tail.len() as isize;
            let lo = (base - max_shift as isize).max(0);
            let hi = (base + max_shift as isize).min(input.len() as isize - prev_tail.len() as isize);
            let mut d = lo;
            while d <= hi {
                let seg = &input[d as usize..d as usize + prev_tail.len()];
                let s: f64 = seg.iter().zip(prev_tail.iter()).map(|(a, b)| (*a as f64) * (*b as f64)).sum();
                if s > best_score {
                    best_score = s;
                    best_d = d - base;
                }
                d += 4; // coarse search step for speed
            }
        }
        let aligned_start = (read as isize + best_d) as usize;
        let start = aligned_start.min(input.len().saturating_sub(frame));
        let seg = &input[start..start + frame];
        for i in 0..frame {
            let o = write + i;
            if o < out.len() {
                out[o] += seg[i] * win[i];
                norm[o] += win[i] * win[i];
            }
        }
        // remember tail of this frame for next similarity search
        let tl = frame / 2;
        prev_tail = seg[frame - tl..].to_vec();
        read = (start + frame / 2) as usize; // advance half frame in input
        write += hop_out;
        if read + frame >= input.len() {
            break;
        }
    }
    // normalize by window power sum to avoid amplitude modulation
    for i in 0..out.len() {
        if norm[i] > 1e-6 {
            out[i] /= norm[i];
        }
    }
    let target_len = (input.len() as f64 * ratio).round() as usize;
    out.truncate(target_len.min(out.len()));
    out
}

/// Phase-vocoder pitch shift by `semitones` (time preserved).
/// Implementation: resample to change pitch+duration, then WSOLA back to the
/// original length — classic "resample + stretch" which is robust and cheap.
pub fn pitch_shift_vocode(input: &[f32], semitones: f64, sample_rate: f64) -> Vec<f32> {
    if semitones.abs() < 1e-3 || input.len() < 8 {
        return input.to_vec();
    }
    let factor = 2.0f64.powf(semitones / 12.0);
    // Step 1: naive resampling changes both pitch and length.
    let out_len = (input.len() as f64 / factor).ceil() as usize;
    let mut resampled = Vec::with_capacity(out_len);
    for i in 0..out_len {
        let pos = i as f64 * factor;
        let i0 = pos.floor() as usize;
        let frac = pos - i0 as f64;
        let a = *input.get(i0).unwrap_or(&0.0) as f64;
        let b = *input.get(i0 + 1).unwrap_or(&0.0) as f64;
        resampled.push((a + (b - a) * frac) as f32);
    }
    // Step 2: WSOLA back to the original duration (pitch stays shifted).
    wsola_stretch(&resampled, factor, sample_rate)
}

/// Estimate artifact risk of a stretch ratio: beyond ±8% WSOLA starts to
/// produce flanging/"robotic" artifacts on vocal material. Used by the
/// planner's artifact penalty term.
pub fn stretch_artifact_risk(ratio: f64) -> f64 {
    let pct = (ratio - 1.0).abs() * 100.0;
    if pct <= 6.0 {
        0.05 * pct / 6.0
    } else if pct <= 8.0 {
        0.05 + 0.25 * (pct - 6.0) / 2.0
    } else {
        (0.3 + 0.7 * ((pct - 8.0) / 10.0)).min(1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wsola_preserves_length_ratio_and_pitch() {
        let fs = 44100.0;
        let n = (fs * 2.0) as usize;
        // 220 Hz tone + periodic pulse every 0.5 s (gives WSOLA anchors)
        let sig: Vec<f32> = (0..n)
            .map(|i| {
                let t = i as f64 / fs;
                let tone = (2.0 * std::f64::consts::PI * 220.0 * t).sin() as f32 * 0.5;
                let pulse = if (i % (fs as usize / 2)) < 64 { 0.5 } else { 0.0 };
                tone + pulse
            })
            .collect();
        let stretched = wsola_stretch(&sig, 1.05, fs);
        assert!(((stretched.len() as f64 / n as f64) - 1.05).abs() < 0.02);
        // dominant frequency should still be ~220 Hz
        let fft = Fft::new(4096);
        let mut ps = vec![0.0; 2049];
        let seg = &stretched[n / 2..n / 2 + 4096.min(stretched.len() - n / 2)];
        fft.power_spectrum(seg, &mut ps);
        let peak = ps.iter().enumerate().skip(1).take(100).max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        let f = peak as f64 * fs / 4096.0;
        assert!((f - 220.0).abs() < 15.0, "pitch moved to {f}");
    }

    #[test]
    fn pitch_shift_moves_frequency() {
        let fs = 44100.0;
        let n = (fs * 2.0) as usize;
        let sig: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / fs).sin() as f32)
            .collect();
        let shifted = pitch_shift_vocode(&sig, 2.0, fs);
        assert!((shifted.len() as f64 - n as f64).abs() < (n as f64 * 0.05));
        let fft = Fft::new(4096);
        let mut ps = vec![0.0; 2049];
        let mid = n / 2;
        let seg = &shifted[mid..mid + 4096.min(shifted.len() - mid)];
        fft.power_spectrum(seg, &mut ps);
        let peak = ps.iter().enumerate().skip(1).take(200).max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0;
        let f = peak as f64 * fs / 4096.0;
        let expect = 440.0 * 2.0f64.powf(2.0 / 12.0);
        assert!((f - expect).abs() < 25.0, "expected ~{expect}, got {f}");
    }

    #[test]
    fn artifact_risk_monotonic() {
        assert!(stretch_artifact_risk(1.0) < stretch_artifact_risk(1.07));
        assert!(stretch_artifact_risk(1.07) < stretch_artifact_risk(1.15));
    }
}
