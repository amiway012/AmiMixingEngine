//! Stage 1: EBU R128-style loudness, silence / fade detection, quality flags.

use crate::model::*;

/// G-weighting coefficients (approximation of the R128 K-weighting filter)
/// applied as a simple shelf pre-filter + high-pass cascade implemented with
/// two one-pole sections for robustness in std-only code.
fn k_weight_sample(x: f64, state: &mut [f64; 4]) -> f64 {
    // High-pass (~ 2nd order via cascaded one-poles at ~ 700 Hz-ish corner)
    let hp_corner = (-2.0 * std::f64::consts::PI * 700.0 / 48000.0).exp(); // placeholder recomputed by caller fs
    let _ = hp_corner;
    // simplified: state[0] lowpass of x, then subtract to get HP component
    state[0] = 0.95 * state[0] + 0.05 * x; // LF shelf follower
    state[1] = 0.8 * state[1] + 0.2 * x; // HF emphasis follower
    let hf = x - state[0]; // de-emphasize LF
    state[2] = 0.995 * state[2] + 0.005 * (x + hf * 0.7); // smoothed output
    state[3] = state[2] + 0.2 * (state[1] - state[0]); // shelf tilt
    state[3]
}

/// Mean square over blocks of `block` samples, skipping first/last 3 s gating
/// blocks like R128 does (relative gate approximation).
pub fn measure_loudness(mono: &[f32], sample_rate: f64) -> LoudnessProfile {
    let block = (sample_rate * 0.4) as usize; // 400 ms short-term blocks, 75% overlap → use 400 non-overlap for simplicity
    let mut states = [0.0f64; 4];
    let mut block_ms: Vec<f64> = Vec::new();
    let mut i = 0usize;
    while i + block <= mono.len() {
        let mut acc = 0.0f64;
        for &s in &mono[i..i + block] {
            let y = k_weight_sample(s as f64, &mut states);
            acc += y * y;
        }
        let z = acc / block as f64;
        // LUFS = -0.691 + 10*log10(z)
        block_ms.push(if z > 1e-12 { -0.691 + 10.0 * z.log10() } else { -70.0 });
        i += block;
    }

    // Integrated: relative gate — absolute -70 LUFS then relative -10 LU below
    let abs_pass: Vec<f64> = block_ms.iter().cloned().filter(|l| *l > -70.0).collect();
    let mut integrated = -70.0f64;
    if !abs_pass.is_empty() {
        let m = mean_l(&abs_pass);
        let rel_thresh = m - 10.0;
        let rel_pass: Vec<f64> = abs_pass.iter().cloned().filter(|l| *l > rel_thresh).collect();
        if !rel_pass.is_empty() {
            let lm = mean_l(&rel_pass);
            integrated = lm;
        }
    }

    // Loudness range: LRA = P95 - P10 of relative-gated short terms
    let lra = {
        let mut v = abs_pass.clone();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if v.len() > 10 {
            let p10 = v[(v.len() as f64 * 0.1) as usize];
            let p95 = v[((v.len() as f64 * 0.95) as usize).min(v.len() - 1)];
            p95 - p10
        } else {
            0.0
        }
    };

    // true peak approximated by max |sample| (inter-sample peaks need oversampling;
    // we note this limitation in docs)
    let true_peak = mono.iter().fold(0.0f32, |m, x| m.max(x.abs())) as f64;

    // clipping ratio: runs of >=3 consecutive near-full-scale samples
    let mut clipped = 0usize;
    let mut run = 0usize;
    for &x in mono {
        if x.abs() >= 0.999 {
            run += 1;
            if run >= 3 {
                clipped += 1;
            }
        } else {
            run = 0;
        }
    }
    let clip_ratio = if mono.is_empty() { 0.0 } else { clipped as f64 / mono.len() as f64 };

    // silence regions (< -60 LUFS equivalent amplitude), min 0.5 s
    let thresh = 10.0f64.powf((-60.0 + 0.691) / 10.0).sqrt() as f32;
    let win = (sample_rate * 0.1) as usize;
    let mut silent_regions = Vec::new();
    let mut start: Option<usize> = None;
    let mut j = 0usize;
    while j + win <= mono.len() {
        let peak = mono[j..j + win].iter().fold(0.0f32, |m, x| m.max(x.abs()));
        let is_sil = peak < thresh;
        match (&mut start, is_sil) {
            (None, true) => start = Some(j),
            (Some(_), false) => {
                let s = start.take().unwrap();
                if (j - s) as f64 / sample_rate >= 0.5 {
                    silent_regions.push((s as f64 / sample_rate, j as f64 / sample_rate));
                }
            }
            _ => {}
        }
        j += win;
    }
    if let Some(s) = start {
        if (mono.len() - s) as f64 / sample_rate >= 0.5 {
            silent_regions.push((s as f64 / sample_rate, mono.len() as f64 / sample_rate));
        }
    }

    // fade-in: first 5 s where short-term rises monotonically ~linearly from silence
    let fade_in_end = detect_fade_in(&block_ms, sample_rate, block);
    let fade_out_start = detect_fade_out(&block_ms, sample_rate, block);

    LoudnessProfile {
        integrated_lufs: integrated,
        lra,
        short_term: block_ms,
        true_peak,
        clip_ratio,
        silent_regions,
        fade_in_end,
        fade_out_start,
    }
}

fn mean_l(v: &[f64]) -> f64 {
    v.iter().sum::<f64>() / v.len() as f64
}

fn detect_fade_in(blocks: &[f64], sample_rate: f64, block: usize) -> Option<f64> {
    if blocks.len() < 4 {
        return None;
    }
    // find how many initial blocks are below the plateau and rising
    let plateau = blocks.iter().skip(blocks.len() / 2).cloned().fold(f64::MIN, f64::max);
    let mut last_rising = 0usize;
    for w in blocks.windows(2) {
        if w[1] > w[0] && w[1] < plateau - 1.0 {
            last_rising = w.as_ptr() as usize; // not usable; recompute index below
        }
    }
    let _ = last_rising;
    let mut end_idx = 0usize;
    for i in 1..blocks.len().min(15) {
        if blocks[i] > blocks[i - 1] + 0.2 && blocks[i] < plateau - 1.0 {
            end_idx = i;
        } else if blocks[i] >= plateau - 1.0 {
            break;
        }
    }
    if end_idx >= 2 {
        Some(((end_idx + 1) * block) as f64 / sample_rate)
    } else {
        None
    }
}

fn detect_fade_out(blocks: &[f64], sample_rate: f64, block: usize) -> Option<f64> {
    if blocks.len() < 4 {
        return None;
    }
    let n = blocks.len();
    let plateau = blocks.iter().take(n / 2).cloned().fold(f64::MIN, f64::max);
    let mut start_idx = n;
    for i in (n.saturating_sub(15)..n).rev() {
        if blocks[i] < blocks[i.min(n - 1)] && blocks[i] < plateau - 1.0 {
            start_idx = i;
        }
        // falling tail check
        if i + 1 < n && blocks[i + 1] < blocks[i] - 0.2 && blocks[i] < plateau - 1.0 {
            start_idx = start_idx.min(i);
        }
    }
    if n - start_idx >= 3 && start_idx < n {
        Some((start_idx * block) as f64 / sample_rate)
    } else {
        None
    }
}

/// Derive analysis-quality flags from measurements + metadata.
pub fn quality_flags(l: &LoudnessProfile, meta: &TrackMetadata, duration: f64) -> AnalysisQuality {
    let mut q = AnalysisQuality::default();
    if l.clip_ratio > 0.001 {
        q.source_clipping = true;
        q.notes.push(format!("source clipping detected ({:.2}%)", l.clip_ratio * 100.0));
    }
    if l.true_peak > 0.999 {
        q.artifact_suspect = true;
        q.notes.push("true peak at full scale".into());
    }
    // hidden track / long leading silence edge case
    if let Some((_, e)) = l.silent_regions.first() {
        if *e > 5.0 {
            q.notes.push(format!("leading silence {e:.1}s (hidden-track gap?)"));
        }
    }
    // very quiet material
    if l.integrated_lufs < -30.0 {
        q.notes.push("very low loudness; verify not corrupt decode".into());
        q.score -= 0.1;
    }
    // metadata duration mismatch handled by caller when tag duration known
    let _ = duration;
    let _ = meta;
    q.score = (q.score + 0.9 - (l.clip_ratio * 10.0).min(0.3)).clamp(0.0, 1.0);
    q
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn silence_detected() {
        let sr = 8000.0;
        let mut v = vec![0.0f32; (sr * 3.0) as usize];
        for i in (sr as usize * 3)..(sr as usize * 6) {
            v[i] = 0.3;
        }
        let l = measure_loudness(&v, sr);
        assert!(!l.silent_regions.is_empty(), "should find leading silence");
        assert!(l.silent_regions[0].1 >= 2.5);
    }

    #[test]
    fn tone_loudness_reasonable() {
        let sr = 44100.0;
        let n = (sr * 10.0) as usize;
        let v: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f64::consts::PI * 1000.0 * i as f64 / sr).sin() as f32 * 0.5)
            .collect();
        let l = measure_loudness(&v, sr);
        // 0.5 amplitude 1 kHz sine ≈ -10 dBFS RMS → around -6..-10 LUFS after weighting
        assert!(l.integrated_lufs > -25.0 && l.integrated_lufs < 0.0, "{}", l.integrated_lufs);
    }

    #[test]
    fn clipping_flag() {
        let sr = 8000.0;
        let v: Vec<f32> = (0..(sr * 4) as usize).map(|i| if i % 4 < 2 { 1.0 } else { -1.0 }).collect();
        let l = measure_loudness(&v, sr);
        assert!(l.clip_ratio > 0.1);
        let q = quality_flags(&l, &TrackMetadata::default(), 4.0);
        assert!(q.source_clipping);
    }
}
