//! Radix-2 iterative FFT (in-place, Cooley-Tukey) with a cached bit-reversal
//! table and precomputed twiddles. Sufficient for 1024–4096 point analysis
//! windows at DJ frame rates.

/// Precomputed plan for one power-of-two size.
pub struct Fft {
    n: usize,
    rev: Vec<usize>,
    /// Twiddle factors e^{-2*pi*i*k/n} as (re, im).
    tw: Vec<(f64, f64)>,
}

impl Fft {
    /// Create a plan; panics if `n` is not a power of two >= 2.
    pub fn new(n: usize) -> Self {
        assert!(n >= 2 && n.is_power_of_two(), "FFT size must be power of two");
        let bits = n.trailing_zeros() as usize;
        let rev: Vec<usize> = (0..n)
            .map(|i| {
                let mut r = 0usize;
                let mut x = i;
                for _ in 0..bits {
                    r = (r << 1) | (x & 1);
                    x >>= 1;
                }
                r
            })
            .collect();
        let mut tw = Vec::with_capacity(n / 2);
        for k in 0..n / 2 {
            let a = -std::f64::consts::PI * k as f64 * 2.0 / n as f64;
            tw.push((a.cos(), a.sin()));
        }
        Self { n, rev, tw }
    }

    pub fn size(&self) -> usize {
        self.n
    }

    /// In-place forward FFT on interleaved re/im buffers.
    pub fn forward(&self, re: &mut [f64], im: &mut [f64]) {
        let n = self.n;
        assert!(re.len() >= n && im.len() >= n);
        // bit-reversal permutation
        for i in 0..n {
            let j = self.rev[i];
            if j > i {
                re.swap(i, j);
                im.swap(i, j);
            }
        }
        // butterflies
        let mut len = 2;
        while len <= n {
            let step = n / len;
            let mut i = 0;
            while i < n {
                for k in 0..len / 2 {
                    let (wr, wi) = self.tw[k * step];
                    let a = i + k;
                    let b = i + k + len / 2;
                    let xr = re[b] * wr - im[b] * wi;
                    let xi = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - xr;
                    im[b] = im[a] - xi;
                    re[a] += xr;
                    im[a] += xi;
                }
                i += len;
            }
            len <<= 1;
        }
    }

    /// Power spectrum magnitude of a real signal (`frame` length <= n, zero-padded).
    /// Returns n/2+1 magnitudes.
    pub fn power_spectrum(&self, frame: &[f32], out: &mut [f64]) {
        let n = self.n;
        let mut re = vec![0.0f64; n];
        let mut im = vec![0.0f64; n];
        let m = frame.len().min(n);
        for i in 0..m {
            re[i] = frame[i] as f64;
        }
        self.forward(&mut re, &mut im);
        let bins = n / 2 + 1;
        debug_assert!(out.len() >= bins);
        for k in 0..bins {
            out[k] = re[k] * re[k] + im[k] * im[k];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sine_peak_at_correct_bin() {
        let n = 1024;
        let fft = Fft::new(n);
        // 100 Hz sine at 44100 => bin = 100 * 1024 / 44100 ≈ 2.32 — use exact bin freq
        let bin = 8usize;
        let f = bin as f64 * 44100.0 / n as f64;
        let frame: Vec<f32> = (0..n)
            .map(|i| (std::f64::consts::TAU * f * i as f64 / 44100.0).sin() as f32)
            .collect();
        let mut ps = vec![0.0; n / 2 + 1];
        fft.power_spectrum(&frame, &mut ps);
        let peak = ps
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(peak, bin);
    }

    #[test]
    fn parseval_energy_preserved() {
        let n = 512;
        let fft = Fft::new(n);
        let frame: Vec<f32> = (0..n).map(|i| ((i as f64 * 0.1).sin() * 0.5) as f32).collect();
        let time_e: f64 = frame.iter().map(|x| (*x as f64).powi(2)).sum();
        let mut ps = vec![0.0; n / 2 + 1];
        fft.power_spectrum(&frame, &mut ps);
        // scale: sum |X|^2 / N == sum |x|^2 (full spectrum); account for mirror bins
        let spec_e: f64 = (ps[0] + ps[n / 2] + 2.0 * ps[1..n / 2].iter().sum::<f64>()) / (n as f64 * n as f64);
        assert!((spec_e - time_e).abs() / time_e < 1e-6);
    }
}
