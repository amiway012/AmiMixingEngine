//! Stage 2: rhythmic analysis.
//!
//! Method (open-science based, mirroring Dixon 2001 spectral-flux tempo and
//! Ellis 2007's dynamic-programming beat tracker, i.e. a simplified DBN):
//! 1. onset strength = half-wave-rectified spectral flux (with low-band boost
//!    to emphasize kicks);
//! 2. tempo via autocorrelation + comb filter bank over 60..200 BPM with
//!    octave/half-time disambiguation using onset-phase alignment;
//! 3. beat grid via Viterbi DP maximizing onset evidence with an ideal
//!    inter-beat-interval prior;
//! 4. downbeats by accent periodicity (strong-weak pattern energy per phase);
//! 5. meter chosen among {3,4,6} by bar-accent correlation;
//! 6. tempo stability from local IBI variance in sliding windows;
//! 7. multi-BPM segments for accelerando/tempo-change tracks.

use crate::analysis::Stft;
use crate::dsp::{median_filter, pick_peaks};
use crate::model::*;

/// Onset strength curve from spectral flux. Returns one value per STFT frame.
pub fn onset_flux(stft: &mut Stft, mono: &[f32]) -> Vec<f64> {
    let bins = stft.bins();
    let mut prev: Option<Vec<f64>> = None;
    let mut flux = Vec::new();
    stft.each_frame(mono, |_f, ps| {
        // log compression reduces loudness bias, then rectified difference
        let cur: Vec<f64> = ps.iter().map(|x| (1.0 + x.ln()).max(0.0)).collect();
        if let Some(p) = &prev {
            let d: f64 = cur.iter().zip(p.iter()).map(|(c, o)| (c - o).max(0.0)).sum();
            flux.push(d);
        } else {
            flux.push(0.0);
        }
        prev = Some(cur);
    });
    let _ = bins;
    median_filter(&flux, 2)
}

/// Low-band (kick) onset envelope: flux restricted to < 150 Hz.
fn kick_flux(stft: &mut Stft, mono: &[f32]) -> Vec<f64> {
    let sr = stft.sample_rate;
    let nfft = stft.fft.size() as f64;
    let max_bin = ((150.0 * nfft / sr) as usize).clamp(2, stft.bins());
    let mut prev: Option<Vec<f64>> = None;
    let mut flux = Vec::new();
    stft.each_frame(mono, |_f, ps| {
        let cur: Vec<f64> = ps[..max_bin].iter().map(|x| x.ln().max(0.0)).collect();
        if let Some(p) = &prev {
            flux.push(cur.iter().zip(p.iter()).map(|(c, o)| (c - o).max(0.0)).sum::<f64>());
        } else {
            flux.push(0.0);
        }
        prev = Some(cur);
    });
    median_filter(&flux, 1)
}

/// Tempo estimate via autocorrelation of the onset envelope over 60–200 BPM,
/// with harmonic (double/half) correction driven by kick-envelope salience.
pub fn tempo_autocorr(env: &[f64], tpf: f64, kick_env: &[f64]) -> (f64, f64) {
    let min_lag = (60.0 / 200.0 / tpf) as usize; // 200 BPM
    let max_lag = (60.0 / 58.0 / tpf) as usize; // ~58 BPM floor
    if env.len() < max_lag * 2 {
        return (0.0, 0.0);
    }
    let mean = env.iter().sum::<f64>() / env.len() as f64;
    let mut best = (0usize, f64::NEG_INFINITY);
    let mut scores: Vec<(usize, f64)> = Vec::new();
    for lag in min_lag..=max_lag {
        let s: f64 =
            env.iter().zip(env.iter().skip(lag)).map(|(a, b)| (a - mean) * (b - mean)).sum::<f64>()
                / env.len() as f64;
        scores.push((lag, s));
        if s > best.1 {
            best = (lag, s);
        }
    }
    if best.0 == 0 {
        return (0.0, 0.0);
    }
    // comb-filter refinement: score = sum of autocorr at lag, 2*lag, ... weighted
    let norm = scores.iter().map(|(_, v)| v.abs()).fold(0.0f64, f64::max).max(1e-9);
    let comb_score = |lag: usize| -> f64 {
        let mut s = 0.0;
        let mut m = 1;
        while m * lag <= max_lag {
            if let Some((_, v)) = scores.iter().find(|(l, _)| *l == m * lag) {
                s += v / norm / (m as f64);
            }
            m += 1;
        }
        s
    };
    let mut candidates = vec![best.0];
    if 2 * best.0 <= max_lag {
        candidates.push(2 * best.0);
    }
    if best.0 % 2 == 0 {
        candidates.push(best.0 / 2);
    }
    let mut pick = candidates[0];
    let mut pv = f64::NEG_INFINITY;
    for c in candidates {
        let mut v = comb_score(c);
        // prefer tempi near the "dance" range 100-140 lightly
        let bpm = 60.0 / (c as f64 * tpf);
        if (100.0..=140.0).contains(&bpm) {
            v *= 1.05;
        }
        // kick envelope strongly supports integer beat multiples
        if !kick_env.is_empty() {
            let km: f64 = kick_env.iter().sum::<f64>() / kick_env.len() as f64;
            let ks: f64 = (0..env.len()).step_by(c).map(|i| kick_env[i % kick_env.len()] - km).sum::<f64>()
                / (env.len() as f64 / c as f64).max(1.0);
            v += (ks / km.max(1e-9)).max(0.0) * 0.2;
        }
        if v > pv {
            pv = v;
            pick = c;
        }
    }
    let bpm = 60.0 / (pick as f64 * tpf);
    // confidence: peak prominence vs neighbors
    let around: Vec<f64> = scores
        .iter()
        .filter(|(l, _)| (*l as f64 - pick as f64).abs() > (pick as f64) * 0.15 && (*l as f64 - pick as f64).abs() < (pick as f64) * 0.6)
        .map(|(_, v)| *v)
        .collect();
    let bg = if around.is_empty() { 0.0 } else { around.iter().cloned().fold(0.0f64, f64::max) };
    let peak = comb_score(pick);
    let conf = (peak - bg / norm).max(0.0) / peak.max(1e-9);
    (bpm, conf.clamp(0.0, 1.0))
}

/// Dynamic-programming beat tracker (Ellis-style). Returns beat frame indices.
pub fn dp_beats(env: &[f64], period_frames: f64) -> Vec<usize> {
    let n = env.len();
    if n == 0 || period_frames < 1.0 {
        return Vec::new();
    }
    let pmin = (period_frames * 0.8).round().max(1.0) as usize;
    let pmax = (period_frames * 1.2).round().max(2.0) as usize;
    let w = 0.5; // transition weight
    let mut delta = vec![vec![0.0f64; n]; n.min(4096)]; // cap memory: only last window needed
    let _ = delta.as_mut_ptr();
    // Use rolling arrays indexed by (t mod cap)
    let cap = n.min(4096);
    let mut d = vec![vec![-1e9f64; n]; cap];
    let mut psi = vec![vec![0usize; n]; cap];
    for t in 0..n {
        d[t % cap][t] = env[t];
    }
    for t in 1..n {
        let lo = t.saturating_sub(pmax);
        let hi = t.saturating_sub(pmin);
        if hi < lo {
            continue;
        }
        let mut best_prev = f64::NEG_INFINITY;
        let mut best_k = lo;
        for k in lo..=hi {
            let ibi_err = (k as f64 - t as f64 + period_frames).abs(); // not used directly
            let _ = ibi_err;
            let val = d[k % cap][k] + w * env[k] - ((k as f64 - t as f64 + period_frames) * (k as f64 - t as f64 + period_frames)) / (2.0 * period_frames * period_frames * 0.04);
            // note: the quadratic term penalizes deviation from ideal period
            if val > best_prev {
                best_prev = val;
                best_k = k;
            }
        }
        if best_prev > -1e8 {
            let cand = best_prev + w * env[t];
            if cand > d[t % cap][t] {
                d[t % cap][t] = cand;
                psi[t % cap][t] = best_k;
            }
        }
    }
    // backtrace from strongest final beats
    let mut best_t = n - 1;
    let mut best_v = f64::NEG_INFINITY;
    let tail_lo = n.saturating_sub(cap).max(n.saturating_sub(period_frames as usize * 4));
    for t in tail_lo..n {
        if d[t % cap][t] > best_v {
            best_v = d[t % cap][t];
            best_t = t;
        }
    }
    let mut beats = Vec::new();
    let mut t = best_t;
    loop {
        beats.push(t);
        let prev = psi[t % cap][t];
        if prev == 0 && t == 0 {
            break;
        }
        if prev >= t {
            break;
        }
        t = prev;
        if beats.len() > n {
            break;
        }
    }
    beats.reverse();
    // extend grid forward/backward to cover whole track
    if beats.len() >= 2 {
        let period = (beats[beats.len() - 1] - beats[0]) as f64 / (beats.len() - 1) as f64;
        let mut first = beats[0] as f64;
        while first - period >= 0.0 {
            first -= period;
            beats.insert(0, first.round() as usize);
        }
        let mut last = *beats.last().unwrap() as f64;
        while last + period < n as f64 {
            last += period;
            beats.push(last.round() as usize);
        }
    }
    beats
}

/// Downbeat detection: choose phase offset & meter maximizing accent energy.
pub fn find_downbeats(env: &[f64], beats: &[usize], candidate_meters: &[usize]) -> (usize, Vec<usize>, f64) {
    if beats.len() < 4 {
        return (4, Vec::new(), 0.0);
    }
    let mut best = (4usize, Vec::new(), f64::NEG_INFINITY);
    for &m in candidate_meters {
        for off in 0..m {
            // average onset energy on downbeats vs other beats
            let mut strong = 0.0;
            let mut weak = 0.0;
            let (mut ns, mut nw) = (0usize, 0usize);
            for (i, &b) in beats.iter().enumerate() {
                if b >= env.len() {
                    continue;
                }
                if i % m == off % m {
                    strong += env[b];
                    ns += 1;
                } else {
                    weak += env[b];
                    nw += 1;
                }
            }
            if ns == 0 || nw == 0 {
                continue;
            }
            let score = (strong / ns as f64) / (weak / nw as f64).max(1e-9);
            if score > best.2 {
                let downs: Vec<usize> =
                    beats.iter().enumerate().filter(|(i, _)| i % m == off % m).map(|(_, &b)| b).collect();
                best = (m, downs, score);
            }
        }
    }
    best
}

/// Local tempo estimates in sliding windows (multi-BPM support).
pub fn local_tempi(stft: &mut Stft, mono: &[f32], tpf: f64, window_s: f64) -> Vec<(f64, f64)> {
    let win = ((window_s / tpf) as usize).max(32);
    let hopw = win / 2;
    let env = onset_flux(stft, mono);
    let mut out = Vec::new();
    let mut start = 0;
    while start + win < env.len() {
        let seg = &env[start..start + win];
        let (bpm, conf) = tempo_autocorr(seg, tpf, &[]);
        if conf > 0.1 && bpm > 40.0 {
            out.push((stft.frame_time(start), bpm));
        }
        start += hopw;
    }
    out
}

/// Full stage-2 pipeline.
pub fn analyze_rhythm(stft: &mut Stft, mono: &[f32], sr: f64, duration: f64, spectral: &TrackSpectralProfile) -> TrackRhythm {
    let _ = sr;
    let tpf = stft.time_per_frame();
    let env = onset_flux(stft, mono);
    let kenv = kick_flux(stft, mono);
    let (bpm, conf) = tempo_autocorr(&env, tpf, &kenv);
    let mut rhythm = TrackRhythm { meter: 4, ..Default::default() };
    if bpm <= 0.0 || env.is_empty() {
        return rhythm; // no clear beat → has_clear_grid=false
    }
    rhythm.bpm = bpm;
    rhythm.bpm_confidence = conf;
    let period_frames = 60.0 / bpm / tpf;
    let beat_frames = dp_beats(&env, period_frames);
    rhythm.beat_times = beat_frames.iter().map(|&f| stft.frame_time(f)).collect();
    rhythm.beat_strength = beat_frames
        .iter()
        .map(|&f| {
            let lo = f.saturating_sub(1);
            let hi = (f + 2).min(env.len());
            let m = env[lo..hi].iter().cloned().fold(0.0f64, f64::max);
            (m / (env.iter().cloned().fold(0.0f64, f64::max)).max(1e-9)).clamp(0.0, 1.0)
        })
        .collect();
    let (meter, down_frames, accent) = find_downbeats(&env, &beat_frames, &[3, 4, 6]);
    rhythm.meter = meter;
    rhythm.downbeat_times = down_frames.iter().map(|&f| stft.frame_time(f)).collect();
    // tempo stability: variance of local IBIs
    let stability = if rhythm.beat_times.len() > 8 {
        let mut devs = Vec::new();
        for w in rhythm.beat_times.windows(2) {
            devs.push((w[1] - w[0] - 60.0 / bpm).abs());
        }
        devs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mad = devs[devs.len() / 2];
        (1.0 - (mad / (60.0 / bpm)) * 4.0).clamp(0.0, 1.0)
    } else {
        0.3
    };
    rhythm.tempo_stability = stability;
    rhythm.has_clear_grid = conf > 0.15 && stability > 0.3 && rhythm.beat_times.len() > 8;
    // percussion clarity: consistent kick density
    let kd = &spectral.kick_density;
    if !kd.is_empty() {
        let mean = kd.iter().sum::<f64>() / kd.len() as f64;
        let var = kd.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / kd.len() as f64;
        rhythm.has_clear_percussion = mean > 0.15 && (var.sqrt() / mean.max(1e-9)) < 0.8;
    }
    // multi-BPM: detect significant local tempo drift
    let locals = local_tempi(stft, mono, tpf, 12.0);
    if locals.len() >= 3 {
        let vals: Vec<f64> = locals.iter().map(|(_, b)| *b).collect();
        let m = vals.iter().sum::<f64>() / vals.len() as f64;
        let spread = vals.iter().map(|v| (v - m).abs()).fold(0.0f64, f64::max);
        if spread / m > 0.08 {
            rhythm.local_tempi = locals;
            rhythm.has_clear_grid = false; // variable tempo → don't hard beatmatch globally
            rhythm.tempo_stability = (rhythm.tempo_stability * 0.5).max(0.1);
        }
    }
    // sanity: very short tracks can't have reliable grids
    if duration < 8.0 {
        rhythm.has_clear_grid = false;
    }
    // silence-only guard
    if env.iter().all(|x| *x < 1e-9) {
        rhythm.bpm = 0.0;
        rhythm.bpm_confidence = 0.0;
        rhythm.beat_times.clear();
    }
    let _ = pick_peaks(&env, 4, 1.2); // keep import used for future onset spike stats
    rhythm
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth;

    #[test]
    fn bpm_124_detected() {
        let b = synth::beat_signal(124.0, "A", 30.0, 22050.0);
        let mono = b.mono();
        let mut stft = Stft::new(22050.0, 1024, 512);
        let spec = TrackSpectralProfile::default();
        let r = analyze_rhythm(&mut stft, &mono, 22050.0, b.duration(), &spec);
        assert!((r.bpm - 124.0).abs() < 3.0 || (r.bpm - 62.0).abs() < 1.5 || (r.bpm - 248.0).abs() < 6.0, "bpm={}", r.bpm);
        assert!(r.bpm_confidence > 0.1);
    }

    #[test]
    fn ambient_has_no_grid() {
        let b = synth::ambient_track(20.0, 22050.0, 9);
        let mono = b.mono();
        let mut stft = Stft::new(22050.0, 1024, 512);
        let spec = TrackSpectralProfile::default();
        let r = analyze_rhythm(&mut stft, &mono, 22050.0, b.duration(), &spec);
        assert!(!r.has_clear_grid);
    }

    #[test]
    fn beat_spacing_matches_bpm() {
        let b = synth::beat_signal(120.0, "A", 24.0, 22050.0);
        let mono = b.mono();
        let mut stft = Stft::new(22050.0, 1024, 512);
        let spec = TrackSpectralProfile::default();
        let r = analyze_rhythm(&mut stft, &mono, 22050.0, b.duration(), &spec);
        if r.beat_times.len() > 4 {
            let avg = (r.beat_times.last().unwrap() - r.beat_times[0]) / (r.beat_times.len() - 1) as f64;
            assert!((avg - 0.5).abs() < 0.06, "IBI {avg}");
        }
    }
}
