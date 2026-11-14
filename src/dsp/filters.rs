//! Biquad filters (RBJ cookbook), one-pole smoothing and a lookahead-free
//! safety limiter used by the mix engine.

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BiquadKind {
    LowPass,
    HighPass,
    BandPass,
    /// Bell EQ around cutoff.
    Peak,
    LowShelf,
    HighShelf,
}

/// Direct-form-I biquad with mutable coefficients.
#[derive(Clone, Debug)]
pub struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl Biquad {
    pub fn new(kind: BiquadKind, fs: f64, freq: f64, q: f64, gain_db: f64) -> Self {
        let mut s = Self { b0: 1.0, b1: 0.0, b2: 0.0, a1: 0.0, a2: 0.0, x1: 0.0, x2: 0.0, y1: 0.0, y2: 0.0 };
        s.set_coeffs(kind, fs, freq, q, gain_db);
        s
    }

    /// Recompute RBJ coefficients. Keeps filter state (click-free automation).
    pub fn set_coeffs(&mut self, kind: BiquadKind, fs: f64, freq: f64, q: f64, gain_db: f64) {
        let f = freq.clamp(10.0, fs * 0.49);
        let w0 = 2.0 * std::f64::consts::PI * f / fs;
        let cw = w0.cos();
        // Q clamped to avoid NaNs on automation extremes
        let q = q.max(0.3).min(20.0);
        let alpha = w0.sin() / (2.0 * q);
        let a0;
        match kind {
            BiquadKind::LowPass => {
                self.b0 = (1.0 - cw) / 2.0;
                self.b1 = 1.0 - cw;
                self.b2 = (1.0 - cw) / 2.0;
                a0 = 1.0 + alpha;
                self.a1 = -2.0 * cw;
                self.a2 = 1.0 - alpha;
            }
            BiquadKind::HighPass => {
                self.b0 = (1.0 + cw) / 2.0;
                self.b1 = -(1.0 + cw);
                self.b2 = (1.0 + cw) / 2.0;
                a0 = 1.0 + alpha;
                self.a1 = -2.0 * cw;
                self.a2 = 1.0 - alpha;
            }
            BiquadKind::BandPass => {
                self.b0 = alpha;
                self.b1 = 0.0;
                self.b2 = -alpha;
                a0 = 1.0 + alpha;
                self.a1 = -2.0 * cw;
                self.a2 = 1.0 - alpha;
            }
            BiquadKind::Peak => {
                let a = 10.0_f64.powf(gain_db / 40.0);
                self.b0 = 1.0 + alpha * a;
                self.b1 = -2.0 * cw;
                self.b2 = 1.0 - alpha * a;
                a0 = 1.0 + alpha / a;
                self.a1 = -2.0 * cw;
                self.a2 = 1.0 - alpha / a;
            }
            BiquadKind::LowShelf => {
                let a = 10.0_f64.powf(gain_db / 40.0);
                let sq = (a * a + 1.0).sqrt();
                self.b0 = a * ((a + 1.0) - (a - 1.0) * cw + 2.0 * a.sqrt() * alpha);
                self.b1 = 2.0 * a * ((a - 1.0) - (a + 1.0) * cw);
                self.b2 = a * ((a + 1.0) - (a - 1.0) * cw - 2.0 * a.sqrt() * alpha);
                a0 = (a + 1.0) + (a - 1.0) * cw + 2.0 * a.sqrt() * alpha;
                self.a1 = -2.0 * ((a - 1.0) + (a + 1.0) * cw);
                self.a2 = (a + 1.0) + (a - 1.0) * cw - 2.0 * a.sqrt() * alpha;
                let _ = sq;
            }
            BiquadKind::HighShelf => {
                let a = 10.0_f64.powf(gain_db / 40.0);
                self.b0 = a * ((a + 1.0) + (a - 1.0) * cw + 2.0 * a.sqrt() * alpha);
                self.b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cw);
                self.b2 = a * ((a + 1.0) + (a - 1.0) * cw - 2.0 * a.sqrt() * alpha);
                a0 = (a + 1.0) - (a - 1.0) * cw + 2.0 * a.sqrt() * alpha;
                self.a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cw);
                self.a2 = (a + 1.0) - (a - 1.0) * cw - 2.0 * a.sqrt() * alpha;
            }
        }
        let inv = 1.0 / a0;
        self.b0 *= inv;
        self.b1 *= inv;
        self.b2 *= inv;
        self.a1 *= inv;
        self.a2 *= inv;
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let y = self.b0 * x as f64 + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1 - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x as f64;
        self.y2 = self.y1;
        self.y1 = y;
        y as f32
    }

    pub fn reset(&mut self) {
        self.x1 = 0.0; self.x2 = 0.0; self.y1 = 0.0; self.y2 = 0.0;
    }
}

/// Simple one-pole lowpass used for envelope following & parameter smoothing
/// (prevents zipper noise on automation).
#[derive(Clone, Debug)]
pub struct OnePole {
    coef: f64,
    y: f64,
}

impl OnePole {
    /// `time_constant_s`: per-sample pole for given tc at `fs`.
    pub fn new(fs: f64, time_constant_s: f64) -> Self {
        let coef = (-1.0 / (fs * time_constant_s.max(1e-6))).exp();
        Self { coef, y: 0.0 }
    }

    #[inline]
    pub fn process(&mut self, x: f64) -> f64 {
        self.y = self.coef * self.y + (1.0 - self.coef) * x;
        self.y
    }

    pub fn set(&mut self, v: f64) {
        self.y = v;
    }
}

/// Soft-knee safety limiter keeping output below a ceiling (spec §5 rule 7).
/// Uses a short peak-hold envelope; attack ~0.5 ms, release ~80 ms.
#[derive(Clone, Debug)]
pub struct Limiter {
    threshold: f64,
    ceiling: f64,
    gain: f64,
    atk: f64,
    rel: f64,
}

impl Limiter {
    pub fn new(fs: f64, ceiling_db: f64) -> Self {
        let ceiling = 10.0_f64.powf(ceiling_db / 20.0) as f64;
        Self {
            threshold: 0.9,
            ceiling,
            gain: 1.0,
            atk: (-1.0 / (fs * 0.0005)).exp(),
            rel: (-1.0 / (fs * 0.08)).exp(),
        }
    }

    #[inline]
    pub fn process(&mut self, x: f32) -> f32 {
        let ax = x.abs() as f64;
        let target = if ax > self.threshold {
            // soft knee: compress above threshold
            let over = ax - self.threshold;
            (self.threshold + over * 0.5 / (1.0 + over)).min(self.ceiling) / ax.max(1e-9)
        } else {
            1.0
        };
        let coef = if target < self.gain { self.atk } else { self.rel };
        self.gain = coef * self.gain + (1.0 - coef) * target;
        let y = x as f64 * self.gain;
        // hard clip at ceiling as last resort (still sample-continuous thanks to smooth gain)
        (y.clamp(-self.ceiling, self.ceiling)) as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowpass_attenuates_high_freq() {
        let fs = 44100.0;
        let mut lp = Biquad::new(BiquadKind::LowPass, fs, 200.0, 0.7);
        // feed 5 kHz sine, measure RMS after settling
        let n = 4096;
        let mut e_in = 0.0;
        let mut e_out = 0.0;
        for i in 0..n {
            let x = (2.0 * std::f64::consts::PI * 5000.0 * i as f64 / fs).sin() as f32;
            let y = lp.process(x);
            e_in += (x as f64).powi(2);
            if i > 512 {
                e_out += (y as f64).powi(2);
            }
        }
        assert!(e_out / e_in < 0.01, "LP did not attenuate");
    }

    #[test]
    fn highpass_removes_dc() {
        let mut hp = Biquad::new(BiquadKind::HighPass, 44100.0, 100.0, 0.7);
        let mut y = 0.0;
        for _ in 0..8192 {
            y = hp.process(1.0);
        }
        assert!(y.abs() < 1e-3);
    }

    #[test]
    fn limiter_holds_ceiling() {
        let mut lim = Limiter::new(44100.0, -0.3);
        let ceil = 10f64.powf(-0.3 / 20.0) as f32;
        let mut max = 0.0f32;
        for i in 0..8192 {
            let x = if i % 100 < 50 { 1.4 } else { -1.4 };
            let y = lim.process(x);
            max = max.max(y.abs());
        }
        assert!(max <= ceil + 1e-4, "exceeded ceiling: {max} vs {ceil}");
    }
}
