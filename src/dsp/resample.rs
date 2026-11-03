//! Sample-rate conversion: windowed-sinc (offline, high quality) and linear
//! (realtime fallback). Used by stage 1 to normalize 44.1/48 kHz etc.

/// High-quality offline resampling with a windowed sinc kernel.
/// `rolloff`<1 avoids aliasing near Nyquist of the lower rate.
pub fn resample_sinc(input: &[f32], src_rate: f64, dst_rate: f64, taps_per_side: usize) -> Vec<f32> {
    if (src_rate - dst_rate).abs() < 1e-6 || input.is_empty() {
        return input.to_vec();
    }
    let ratio = dst_rate / src_rate;
    let rolloff = 0.92;
    let cutoff = ratio.min(1.0) * rolloff;
    let scale = cutoff * std::f64::consts::PI / ratio.max(1.0).min(1.0); // per-input-sample phase step
    let out_len = (input.len() as f64 * ratio).ceil() as usize;
    let mut out = Vec::with_capacity(out_len);
    let beta = 6.0; // Kaiser-ish fixed window via Tukey approximation
    for i in 0..out_len {
        let t = i as f64 / ratio; // position in input samples
        let center = t.floor() as isize;
        let mut acc = 0.0f64;
        let mut wsum = 0.0f64;
        let lo = center - taps_per_side as isize;
        let hi = center + taps_per_side as isize;
        for j in lo..=hi {
            if j < 0 || j as usize >= input.len() {
                continue;
            }
            let d = t - j as f64;
            let x = d * scale;
            let sinc = if x.abs() < 1e-9 { 1.0 } else { (x).sin() / x };
            // Hann window over kernel support
            let u = (d / taps_per_side as f64).clamp(-1.0, 1.0);
            let win = 0.5 * (1.0 + (std::f64::consts::PI * u * 0.5).cos());
            let _ = beta;
            let w = sinc * win;
            acc += input[j as usize] as f64 * w;
            wsum += w;
        }
        out.push(if wsum.abs() > 1e-12 { (acc / wsum) as f32 } else { 0.0 });
    }
    out
}

/// Linear interpolation resampler working on fractional read positions —
/// used by the realtime variable-speed reader (tempo ratio).
pub fn resample_linear(input: &[f32], pos: f64) -> f32 {
    if input.is_empty() {
        return 0.0;
    }
    let p = pos.max(0.0);
    let i0 = p.floor() as usize;
    if i0 + 1 >= input.len() {
        return input[input.len() - 1];
    }
    let frac = p - i0 as f64;
    let a = input[i0] as f64;
    let b = input[i0 + 1] as f64;
    // cubic-ish smoothing of the fraction reduces clickiness slightly
    let s = frac * frac * (3.0 - 2.0 * frac) * 0.5 + frac * 0.5;
    (a + (b - a) * s) as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sinc_preserves_low_tone_energy_ratio() {
        let fs = 44100.0;
        let sig: Vec<f32> = (0..8192)
            .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / fs).sin() as f32)
            .collect();
        let out = resample_sinc(&sig, fs, 48000.0, 32);
        assert_eq!(out.len(), (8192.0 * 48000.0 / 44100.0).ceil() as usize);
        // amplitude should stay close to 1
        let peak = out.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        assert!(peak > 0.95 && peak <= 1.05);
    }

    #[test]
    fn linear_interpolation_midpoint() {
        let buf = [0.0f32, 1.0];
        assert!((resample_linear(&buf, 0.5) - 0.5).abs() < 0.01);
    }
}
