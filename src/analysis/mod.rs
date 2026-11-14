//! Stage 1–5 analysis pipeline (spec sections "Этап 1..5").
//!
//! `AnalysisEngine::analyze` runs, in order:
//! loudness/silence/fade measurement → spectral band profile → rhythm
//! (onset flux + autocorrelation/comb + dynamic-programming beat tracker) →
//! harmony (HPCP chroma + Krumhansl-Schmuckler key + modulation scan) →
//! structure (novelty curve segmentation with phrase alignment).

pub mod beats;
pub mod chroma;
pub mod loudness;
pub mod spectral;
pub mod structure;

use crate::dsp::fft::Fft;
use crate::dsp::hann;
use crate::model::*;

/// Shared STFT state reused across stages (one FFT plan per window size).
pub struct Stft {
    pub fft: Fft,
    pub window: Vec<f32>,
    pub hop: usize,
    pub sample_rate: f64,
}

impl Stft {
    pub fn new(sample_rate: f64, n_fft: usize, hop: usize) -> Self {
        Self {
            fft: Fft::new(n_fft),
            window: hann(n_fft),
            hop,
            sample_rate,
        }
    }

    pub fn bins(&self) -> usize {
        self.fft.size() / 2 + 1
    }

    /// Number of frames for a mono signal length.
    pub fn n_frames(&self, n_samples: usize) -> usize {
        if n_samples < self.fft.size() {
            return 0;
        }
        (n_samples - self.fft.size()) / self.hop + 1
    }

    /// Compute the power spectrum of frame `f`, writing into `out`.
    pub fn frame_power_into(&mut self, mono: &[f32], f: usize, out: &mut [f64]) {
        let start = f * self.hop;
        let size = self.fft.size();
        let mut frame = vec![0.0f32; size];
        let avail = mono.len().saturating_sub(start).min(size);
        for i in 0..avail {
            frame[i] = mono[start + i] * self.window[i];
        }
        self.fft.power_spectrum(&frame, out);
    }

    /// Number of analysis frames that would be produced for a signal of `n` samples.
    pub fn num_frames(&self, n: usize) -> usize {
        self.n_frames(n)
    }

    /// Compute the power spectrum of frame `f` into `out` (length >= bins()).
    pub fn frame_spectrum(&mut self, mono: &[f32], f: usize, out: &mut [f64]) {
        self.frame_power_into(mono, f, out);
    }

    /// Iterate all frames calling `cb(frame_idx, power_spectrum)`.
    pub fn each_frame<F: FnMut(usize, &[f64])>(&mut self, mono: &[f32], mut cb: F) {
        let nf = self.num_frames(mono.len());
        let mut ps = vec![0.0f64; self.bins()];
        for f in 0..nf {
            self.frame_power_into(mono, f, &mut ps);
            cb(f, &ps);
        }
    }

    /// Caller-supplied scratch buffer variant (avoids per-call allocation).
    pub fn each_frame_borrowed<F: FnMut(usize, &[f64])>(&mut self, mono: &[f32], ps: &mut [f64], mut cb: F) {
        let nf = self.num_frames(mono.len());
        for f in 0..nf {
            self.frame_power_into(mono, f, ps);
            cb(f, ps);
        }
    }

    /// Time of frame center in seconds.
    pub fn frame_time(&self, f: usize) -> f64 {
        (f * self.hop + self.fft.size() / 2) as f64 / self.sample_rate
    }

    pub fn time_per_frame(&self) -> f64 {
        self.hop as f64 / self.sample_rate
    }
}

/// Engine configuration.
#[derive(Clone, Debug)]
pub struct AnalysisConfig {
    /// Window size for the analysis STFT.
    pub n_fft: usize,
    /// Hop in samples for the analysis STFT (~11.6 ms at 1024/44100).
    pub hop: usize,
    /// If metadata disagrees with audio, prefer audio but flag it.
    pub trust_audio_over_metadata: bool,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self { n_fft: 2048, hop: 512, trust_audio_over_metadata: true }
    }
}

/// The offline analysis engine implementing `api::AudioAnalyzer`.
pub struct AnalysisEngine {
    pub config: AnalysisConfig,
}

impl Default for AnalysisEngine {
    fn default() -> Self {
        Self { config: AnalysisConfig::default() }
    }
}

impl AnalysisEngine {
    /// Full pipeline on a decoded buffer (stages 1–5).
    pub fn analyze(&mut self, buffer: &AudioBuffer) -> TrackAnalysis {
        self.analyze_with_meta(buffer, TrackMetadata { id: "track".into(), ..Default::default() })
    }

    /// Full pipeline including metadata cross-checking.
    pub fn analyze_with_meta(&mut self, buffer: &AudioBuffer, meta: TrackMetadata) -> TrackAnalysis {
        let mono = buffer.mono();
        let sr = buffer.sample_rate;
        let duration = buffer.duration();

        // ---- stage 1: loudness / silence / fades / quality -----------------
        let loudness = loudness::measure_loudness(&mono, sr);
        let mut quality = loudness::quality_flags(&loudness, &meta, duration);

        // ---- stage 5 (computed early; reused by other stages) --------------
        let mut stft = Stft::new(sr, self.config.n_fft.max(256).next_power_of_two(), self.config.hop);
        let spectral = spectral::compute_profile(&mut stft, &mono, &loudness);

        // ---- stage 2: rhythm -------------------------------------------------
        let rhythm = beats::analyze_rhythm(&mut stft, &mono, sr, duration, &spectral);

        // ---- stage 3: harmony ----------------------------------------------
        let harmony = chroma::analyze_harmony(&mut stft, &mono, sr, duration, &rhythm);

        // ---- stage 4: structure ---------------------------------------------
        let structure =
            structure::analyze_structure(&mut stft, &mono, sr, duration, &rhythm, &harmony, &spectral, &loudness);

        // metadata vs audio consistency (edge case: wrong tags)
        if let Some(dbpm) = meta.declared_bpm {
            if rhythm.has_clear_grid && (dbpm - rhythm.bpm).abs() > 6.0 {
                quality.notes.push(format!(
                    "metadata BPM {dbpm:.1} differs from analyzed {:.1}; using audio",
                    rhythm.bpm
                ));
                if !self.config.trust_audio_over_metadata {
                    quality.score *= 0.7;
                }
            }
        }

        TrackAnalysis {
            id: meta.id.clone(),
            duration,
            sample_rate: sr,
            metadata: meta,
            rhythm,
            harmony,
            structure,
            spectral,
            loudness,
            quality,
        }
    }
}
