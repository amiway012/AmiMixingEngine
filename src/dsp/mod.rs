//! Low-level DSP building blocks: FFT, windowing, STFT, biquad filters,
//! mel/HPCP helpers, WSOLA time stretching, phase-vocoder pitch shifting and
//! a safety limiter. Everything is std-only and allocation-friendly for RT use.

pub mod fft;
pub mod filters;
pub mod resample;
pub mod stretch;

pub use fft::Fft;
pub use filters::{Biquad, BiquadKind, OnePole};
pub use resample::resample_linear;
pub use stretch::{pitch_shift_vocode, wsola_stretch};

use crate::model::CurveShape;

/// Hann window table (cached per size via caller).
pub fn hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            0.5 * (1.0 - (std::f64::consts::TAU * i as f64 / (n as f64 - 1.0)).cos()) as f32
        })
        .collect()
}

/// Root-Raised-Cosine-ish power window used by WSOLA (equal energy overlap).
pub fn overlapping_cos_window(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let x = std::f64::consts::PI * i as f64 / n as f64;
            // sin^2 => complementary with itself when overlapped at 50%
            (x.sin() * x.sin()) as f32
        })
        .collect()
}

/// Equal-power crossfade gains for a normalized position 0..1.
/// Returns (gain_a, gain_b) such that gain_a^2 + gain_b^2 == 1.
#[inline]
pub fn equal_power_gains(x: f64) -> (f64, f64) {
    let x = x.clamp(0.0, 1.0);
    let a = ((1.0 - x) * std::f64::consts::FRAC_PI_2).cos();
    let b = ((1.0 - x) * std::f64::consts::FRAC_PI_2).sin();
    (a, b)
}

/// Crossfade gains according to a curve shape.
#[inline]
pub fn curve_gains(x: f64, shape: CurveShape) -> (f64, f64) {
    match shape {
        CurveShape::EqualPower => equal_power_gains(x),
        CurveShape::Linear => {
            let x = x.clamp(0.0, 1.0);
            (1.0 - x, x)
        }
        CurveShape::Expo => {
            let x = x.clamp(0.0, 1.0);
            let e = x * x;
            (1.0 - e, e.sqrt().max(x))
        }
        CurveShape::SCurve => {
            let x = x.clamp(0.0, 1.0);
            let s = x * x * (3.0 - 2.0 * x);
            equal_power_gains(s)
        }
    }
}

/// Mean of a slice.
pub fn mean(v: &[f32]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().map(|x| *x as f64).sum::<f64>() / v.len() as f64
}

/// RMS of a slice.
pub fn rms(v: &[f32]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    (v.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / v.len() as f64).sqrt()
}

/// Normalize a vector to [0,1] by its max.
pub fn normalize_max(v: &mut [f64]) {
    let m = v.iter().cloned().fold(f64::MIN, f64::max);
    if m > 1e-12 {
        for x in v.iter_mut() {
            *x /= m;
        }
    }
}

/// Median filter (window `k*2+1`) — used to smooth novelty/onset curves.
pub fn median_filter(v: &[f64], k: usize) -> Vec<f64> {
    let n = v.len();
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        let lo = i.saturating_sub(k);
        let hi = (i + k + 1).min(n);
        let mut w: Vec<f64> = v[lo..hi].to_vec();
        w.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        out.push(w[w.len() / 2]);
    }
    out
}

/// Peak-pick a positive envelope with adaptive threshold and minimum spacing.
/// Returns indices into `env`.
pub fn pick_peaks(env: &[f64], min_spacing: usize, rel_thresh: f64) -> Vec<usize> {
    let mut peaks = Vec::new();
    if env.is_empty() {
        return peaks;
    }
    let mean = env.iter().sum::<f64>() / env.len() as f64;
    let thresh = mean * rel_thresh;
    let mut i = 1usize;
    let last = env.len() - 1;
    while i < last {
        if env[i] > thresh && env[i] >= env[i - 1] && env[i] > env[i + 1] {
            // local maximum — keep the strongest within spacing
            if let Some(&prev) = peaks.last() {
                if i - prev < min_spacing {
                    if env[i] > env[prev] {
                        *peaks.last_mut().unwrap() = i;
                    }
                    i += 1;
                    continue;
                }
            }
            peaks.push(i);
            i += min_spacing;
            continue;
        }
        i += 1;
    }
    peaks
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equal_power_constant_power() {
        let mut err = 0.0f64;
        for i in 0..=100 {
            let x = i as f64 / 100.0;
            let (a, b) = equal_power_gains(x);
            err = err.max((a * a + b * b - 1.0).abs());
        }
        assert!(err < 1e-9, "equal power violated: {err}");
    }

    #[test]
    fn hann_endpoints_zero() {
        let w = hann(64);
        assert!(w[0].abs() < 1e-6 && w[63].abs() < 1e-6);
    }

    #[test]
    fn peak_pick_finds_periodic_spikes() {
        let mut env = vec![0.0; 400];
        for i in (10..400).step_by(50) {
            env[i] = 1.0;
        }
        let p = pick_peaks(&env, 20, 0.5);
        assert_eq!(p.len(), 8);
    }
}
