//! # automix — fully automatic DJ transition engine
//!
//! A dependency-free (std-only) Rust engine that, given two or more tracks,
//! automatically produces a smooth, musically meaningful transition while
//! minimizing silence gaps, rhythmic conflicts, tonal conflicts and low-frequency mud.
//!
//! ## Pipeline
//!
//! ```text
//! PCM / metadata
//!   -> [decode]    AudioBuffer, resample, loudness (EBU R128), silence/fade detection
//!   -> [rhythm]    onset/spectral-flux -> autocorrelation BPM -> beat grid -> downbeats
//!   -> [harmony]   HPCP chroma -> Krumhansl-Schmuckler key -> Camelot wheel
//!   -> [structure] novelty curve + self-similarity -> segments, phrase boundaries,
//!                   safe entry/exit points
//!   -> [spectral]  5-band energy profile, RMS, centroid, bass-conflict risk
//!   -> [plan]      candidate transitions scored by a weighted cost function
//!   -> [mix]       equal-power crossfade, EQ automation, bass swap, echo out,
//!                   WSOLA time stretch, pitch shift, limiter (ceiling -0.3 dBFS)
//!   -> [rt]        realtime graph with lookahead scheduling
//! ```

pub mod prelude;

pub mod analysis;
pub mod api;
pub mod dsp;
pub mod edge;
pub mod metrics;
pub mod mix;
pub mod model;
pub mod plan;
pub mod rt;
pub mod synth;

pub use analysis::AnalysisEngine;
pub use api::{AudioAnalyzer, AutoMixControllerApi, MixEngineApi, TransitionPlannerApi};
pub use edge::EdgeCaseHandler;
pub use metrics::{BenchmarkReport, QualityMetrics};
pub use model::*;
pub use plan::TransitionPlanner;
