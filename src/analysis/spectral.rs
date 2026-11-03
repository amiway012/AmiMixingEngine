//! Stage 5: five-band spectral energy profile, RMS, centroid, bass-conflict
//! risk, kick transient density and vocal presence per analysis window.

use crate::analysis::Stft;
use crate::dsp::pick_peaks;
use crate::model::*;

/// Compute the full spectral profile from STFT frames.
pub fn compute_profile(stft: &mut Stft, mono: &[f32], _loudness: &LoudnessProfile) -> TrackSpectralProfile {
    let sr = stft.sample_rate;
    let nfft = stft.fft.size();
    let bins = stft.bins();
    let bin_of = |f: f64| -> usize {
        ((f * nfft as f64 / sr).floor() as usize).clamp(0, bins - 1)
    };
    let edges: Vec<(usize, usize)> = BAND_EDGES
        .windows(2)
        .map(|w| (bin_of(w[0]), bin_of(w[1]).max(bin_of(w[0]) + 1)))
        .collect();

    let mut time = Vec::new();
    let mut bands = vec![Vec::new(); 5];
    let mut rms = Vec::new();
    let mut centroid = Vec::new();
    let mut flux_low = Vec::new();
    let mut prev_low: Option<Vec<f64>> = None;

    stft.each_frame(mono, |f, ps| {
        time.push(stft.frame_time(f));
        let total: f64 = ps.iter().sum::<f64>().max(1e-12);
        for (bi, (lo, hi)) in edges.iter().enumerate() {
            let e: f64 = ps[*lo..(*hi).min(bins)].iter().sum();
            bands[bi].push(e / total);
        }
        // frame RMS from the time-domain slice
        let start = f * stft.hop;
        let end = (start + nfft).min(mono.len());
        let mut acc = 0.0f64;
        for &x in &mono[start..end] {
            acc += (x as f64) * (x as f64);
        }
        rms.push((acc / (end - start).max(1) as f64).sqrt());
        let cen: f64 =
            ps.iter().enumerate().map(|(k, v)| k as f64 * v).sum::<f64>() / total * sr / nfft as f64;
        centroid.push(cen);
        // low-band flux for kick transients
        let (lo, hi) = edges[0];
        let cur: Vec<f64> = ps[lo..hi.min(bins)].to_vec();
        if let Some(p) = &prev_low {
            flux_low.push(cur.iter().zip(p.iter()).map(|(c, o)| (c - o).max(0.0)).sum::<f64>());
        } else {
            flux_low.push(0.0);
        }
        prev_low = Some(cur);
    });

    let hop = stft.time_per_frame();
    // normalize each band by its own max → relative shape
    for b in bands.iter_mut() {
        let m = b.iter().cloned().fold(0.0f64, f64::max);
        if m > 1e-12 {
            for x in b.iter_mut() {
                *x /= m;
            }
        }
    }
    // bass conflict risk: band share × absolute loudness
    let max_rms = rms.iter().cloned().fold(0.0f64, f64::max).max(1e-9);
    let bass_conflict_risk: Vec<f64> = bands[0]
        .iter()
        .zip(rms.iter())
        .map(|(share, r)| (share * (r / max_rms)).min(1.0))
        .collect();
    // kick density: peaks of low flux per 0.5 s bucket, replicated to frames
    let peaks = pick_peaks(&flux_low, ((0.2 / hop).max(1.0)) as usize, 1.3);
    let bucket = ((0.5 / hop).max(1.0)) as usize;
    let mut counts = vec![0usize; (flux_low.len() / bucket).max(1)];
    for p in peaks {
        counts[(p / bucket).min(counts.len() - 1)] += 1;
    }
    let cmax = (*counts.iter().max().unwrap_or(&1)).max(1) as f64;
    let kick_density: Vec<f64> = (0..time.len())
        .map(|i| {
            let a = counts[i / bucket].max(counts[((i / bucket) + 1) % counts.len()]);
            a as f64 / cmax
        })
        .collect();
    // vocal presence heuristic: mid/high-mid dominance with moderate centroid
    let vocal_presence: Vec<f64> = (0..time.len())
        .map(|i| {
            let mid_share = bands[2][i] + bands[3][i] * 0.6;
            let low_share = bands[0][i];
            let cent_ok = (centroid[i] > 300.0 && centroid[i] < 4500.0) as u8 as f64;
            (mid_share * 1.2 - low_share * 0.8).max(0.0).min(1.0) * cent_ok
        })
        .collect();

    TrackSpectralProfile {
        time,
        low: bands[0].clone(),
        low_mid: bands[1].clone(),
        mid: bands[2].clone(),
        high_mid: bands[3].clone(),
        high: bands[4].clone(),
        rms,
        centroid,
        bass_conflict_risk,
        kick_density,
        vocal_presence,
        hop,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::analysis::loudness::measure_loudness;
    use crate::synth;

    #[test]
    fn heavy_bass_raises_low_band() {
        let sr = 22050.0;
        let mut o = synth::SynthOptions { bpm: 120.0, duration_s: 12.0, sample_rate: sr, ..Default::default() };
        o.heavy_bass = true;
        let with = synth::synth_track(&o);
        o.heavy_bass = false;
        let without = synth::synth_track(&o);
        let mut st = Stft::new(sr, 1024, 512);
        let l1 = measure_loudness(&with.mono(), sr);
        let p1 = compute_profile(&mut st, &with.mono(), &l1);
        let l2 = measure_loudness(&without.mono(), sr);
        let p2 = compute_profile(&mut st, &without.mono(), &l2);
        assert!(!p1.time.is_empty());
        // with-bass track must have higher absolute low energy
        let abs1: f64 = p1.low.iter().zip(p1.rms.iter()).map(|(s, r)| s * r).sum();
        let abs2: f64 = p2.low.iter().zip(p2.rms.iter()).map(|(s, r)| s * r).sum();
        assert!(abs1 > abs2, "{abs1} vs {abs2}");
    }
}
