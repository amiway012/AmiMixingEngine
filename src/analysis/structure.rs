//! Stage 4: structural segmentation.
//!
//! Novelty curve = frame-to-frame distance of a fused feature vector
//! (chroma + band energies), smoothed and peak-picked with a minimum segment
//! length; boundaries are snapped to bar lines so transitions can align to
//! 4/8/16/32-bar phrases. Segments get labels from energy/bass/vocal/kick
//! statistics relative to track averages.

use crate::analysis::Stft;
use crate::dsp::median_filter;
use crate::model::*;

/// Compute novelty curve over STFT frames.
pub fn novelty(stft: &mut Stft, mono: &[f32], spectral: &TrackSpectralProfile) -> Vec<f64> {
    let sr = stft.sample_rate;
    let nfft = stft.fft.size();
    let mut prev: Option<[f64; 17]> = None; // 12 chroma + 5 bands
    let mut nov = Vec::new();
    stft.each_frame(mono, |f, ps| {
        let mut v = [0.0f64; 17];
        let c = crate::analysis::chroma::chroma_from_spectrum(ps, sr, nfft);
        for i in 0..12 {
            v[i] = c[i];
        }
        if !spectral.low.is_empty() {
            let idx = spectral.idx_at(stft.frame_time(f));
            v[12] = spectral.low[idx];
            v[13] = spectral.low_mid[idx];
            v[14] = spectral.mid[idx];
            v[15] = spectral.high_mid[idx];
            v[16] = spectral.high[idx];
        }
        if let Some(p) = &prev {
            let d: f64 = v.iter().zip(p.iter()).map(|(a, b)| (a - b).abs()).sum();
            nov.push(d);
        } else {
            nov.push(0.0);
        }
        prev = Some(v);
    });
    median_filter(&nov, 3)
}

fn seg_stats(spectral: &TrackSpectralProfile, a: f64, b: f64) -> (f64, f64, f64, f64, f64, f64) {
    if spectral.time.is_empty() {
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    }
    let ia = spectral.idx_at(a);
    let ib = spectral.idx_at(b);
    let lo = ia.min(ib);
    let hi = ia.max(ib).max(lo + 1).min(spectral.time.len());
    let avg = |v: &[f64]| -> f64 {
        if v.is_empty() || hi <= lo {
            return 0.0;
        }
        v[lo..hi].iter().sum::<f64>() / (hi - lo) as f64
    };
    (
        avg(&spectral.rms),
        avg(&spectral.low),
        avg(&spectral.mid),
        avg(&spectral.high),
        avg(&spectral.vocal_presence),
        avg(&spectral.kick_density),
    )
}

/// Label a segment given its position and stats relative to the track.
#[allow(clippy::too_many_arguments)]
fn label_of(
    pos: f64,
    dur: f64,
    len: f64,
    energy: f64,
    mean_energy: f64,
    bass: f64,
    vocal: f64,
    kick: f64,
    is_silent: bool,
) -> SegmentLabel {
    if is_silent {
        return SegmentLabel::SilentGap;
    }
    let rel = energy / mean_energy.max(1e-9);
    if pos < 0.12 * dur && len < 25.0 && rel < 1.15 {
        return SegmentLabel::Intro;
    }
    if pos + len > dur * 0.88 {
        if rel < 0.5 {
            return SegmentLabel::Tail;
        }
        return SegmentLabel::Outro;
    }
    if rel > 1.25 && kick > 0.5 {
        return SegmentLabel::Drop;
    }
    if rel > 1.1 {
        return SegmentLabel::Chorus;
    }
    if rel < 0.55 && bass < 0.35 {
        return if kick > 0.55 { SegmentLabel::BuildUp } else { SegmentLabel::Breakdown };
    }
    if vocal > 0.5 {
        return SegmentLabel::Vocal;
    }
    if vocal < 0.2 && (0.55..=1.1).contains(&rel) {
        return SegmentLabel::Instrumental;
    }
    SegmentLabel::Verse
}

/// Full stage-4 pipeline.
pub fn analyze_structure(
    stft: &mut Stft,
    mono: &[f32],
    _sr: f64,
    duration: f64,
    rhythm: &TrackRhythm,
    harmony: &TrackHarmony,
    spectral: &TrackSpectralProfile,
    loudness: &LoudnessProfile,
) -> TrackStructure {
    let tpf = stft.time_per_frame();
    let nf = stft.n_frames(mono.len());
    let mut out = TrackStructure { from_audio: true, ..Default::default() };
    if nf < 8 || duration <= 0.0 {
        return out;
    }
    let nov = novelty(stft, mono, spectral);
    let bar_len = if rhythm.bpm > 0.0 {
        (60.0 / rhythm.bpm) * rhythm.meter.max(1) as f64
    } else {
        4.0
    };
    let min_seg = (bar_len * 2.0).max(4.0);
    // adaptive threshold on novelty peaks
    let thresh = nov.iter().sum::<f64>() / nov.len() as f64 * 1.15;
    let spacing_frames = ((min_seg / tpf) as usize).max(4);
    let mut peaks: Vec<f64> = Vec::new();
    let mut i = 1usize;
    while i + 1 < nov.len() {
        if nov[i] > thresh && nov[i] >= nov[i - 1] && nov[i] > nov[i + 1] {
            let t = i as f64 * tpf;
            if peaks.last().map_or(true, |&last| t - last >= min_seg * 0.75) {
                peaks.push(t);
                i += spacing_frames;
                continue;
            }
        }
        i += 1;
    }
    // snap to bar grid anchored at first downbeat
    let anchor = rhythm.downbeat_times.first().copied().unwrap_or(0.0);
    let snapped: Vec<f64> = peaks
        .iter()
        .map(|t| anchor + ((t - anchor) / bar_len).round() * bar_len)
        .filter(|t| *t > bar_len && *t < duration - bar_len * 0.5)
        .collect();
    let mut dedup: Vec<f64> = Vec::new();
    for t in snapped {
        if dedup.last().map_or(true, |l| (t - l) > min_seg * 0.5) {
            dedup.push(t);
        }
    }
    let mean_energy = if spectral.rms.is_empty() {
        0.3
    } else {
        spectral.rms.iter().sum::<f64>() / spectral.rms.len() as f64
    };
    let mut bounds = vec![0.0];
    bounds.extend(dedup.iter().cloned());
    bounds.push(duration);
    for w in bounds.windows(2) {
        let (a, b) = (w[0], w[1]);
        let (energy, bass, mid, high, vocal, kick) = seg_stats(spectral, a, b);
        let is_silent = loudness
            .silent_regions
            .iter()
            .any(|(s, e)| a >= s - 0.5 && b <= e + 0.5)
            || energy < mean_energy * 0.05;
        let label = label_of(a, duration, b - a, energy, mean_energy, bass, vocal, kick, is_silent);
        let beat_strength = if rhythm.beat_times.is_empty() || rhythm.bpm <= 0.0 {
            0.0
        } else {
            let cnt = rhythm.beat_times.iter().filter(|t| **t >= a && **t < b).count() as f64;
            let expect = ((b - a) / (60.0 / rhythm.bpm)).max(1.0);
            (cnt / expect).min(1.0)
        };
        out.segments.push(Segment {
            start: a,
            end: b,
            label,
            energy: energy / mean_energy.max(1e-9),
            bass_energy: bass,
            mid_energy: mid,
            high_energy: high,
            vocal_probability: vocal,
            beat_strength,
            key_confidence: harmony.confidence,
        });
    }
    // phrase boundaries every 4 bars aligned to downbeats
    let phrase = bar_len * 4.0;
    let mut p = anchor;
    while p < duration {
        if p >= 0.0 {
            out.phrase_boundaries.push(p);
        }
        p += phrase;
    }
    if out.phrase_boundaries.is_empty() {
        out.phrase_boundaries.push(0.0);
    }
    // safe entry points: phrase boundaries after fade-in/silence with real audio
    let fade_end = loudness.fade_in_end.unwrap_or(0.0);
    for &cand in &out.phrase_boundaries {
        if cand >= fade_end && !is_silent_at(spectral, cand) {
            out.safe_entry_points.push(cand);
        }
        if out.safe_entry_points.len() >= 8 {
            break;
        }
    }
    if out.safe_entry_points.is_empty() {
        out.safe_entry_points.push(fade_end.max(bar_len).min(duration));
    }
    // safe exit points: before fade-out/tail/silence
    let fade_start = loudness.fade_out_start.unwrap_or(duration);
    for &cand in out.phrase_boundaries.iter().rev() {
        if cand <= fade_start && !is_silent_at(spectral, cand) {
            out.safe_exit_points.push(cand);
        }
        if out.safe_exit_points.len() >= 8 {
            break;
        }
    }
    out.safe_exit_points.sort_by(|a, b| b.partial_cmp(a).unwrap());
    if out.safe_exit_points.is_empty() {
        out.safe_exit_points.push(fade_start.min(duration));
    }
    out
}

fn is_silent_at(spectral: &TrackSpectralProfile, t: f64) -> bool {
    if spectral.rms.is_empty() {
        return false;
    }
    spectral.rms[spectral.idx_at(t)] < 0.01
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::{beats, chroma, loudness, spectral as sp};
    use crate::synth;

    #[test]
    fn segments_cover_track_and_have_phrases() {
        let b = synth::beat_signal(122.0, "A", 40.0, 22050.0);
        let mono = b.mono();
        let mut st = Stft::new(22050.0, 1024, 512);
        let l = loudness::measure_loudness(&mono, 22050.0);
        let spec = sp::compute_profile(&mut st, &mono, &l);
        let r = beats::analyze_rhythm(&mut st, &mono, 22050.0, b.duration(), &spec);
        let h = chroma::analyze_harmony(&mut st, &mono, 22050.0, b.duration(), &r);
        let s = analyze_structure(&mut st, &mono, 22050.0, b.duration(), &r, &h, &spec, &l);
        assert!(!s.segments.is_empty());
        assert!((s.segments.first().unwrap().start).abs() < 1e-6);
        assert!((s.segments.last().unwrap().end - b.duration()).abs() < 1.0);
        assert!(!s.phrase_boundaries.is_empty());
        assert!(!s.safe_entry_points.is_empty());
        assert!(!s.safe_exit_points.is_empty());
    }
}
