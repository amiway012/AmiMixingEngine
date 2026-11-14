//! Trait interfaces mirroring the spec section 7 API design.
//!
//! These are the stable boundaries a host application (desktop DJ app, mobile
//! SDK, web via WASM) should program against. Concrete engines implement them;
//! hosts never touch internals.

use crate::model::*;

/// Stream of analysis updates while audio is still being decoded/received —
/// corresponds to `analyzeStream(stream: AudioStream): AnalysisStream`.
pub struct AnalysisStream<'a> {
    chunks: Vec<&'a [f32]>,
    sample_rate: f64,
}

impl<'a> AnalysisStream<'a> {
    pub fn new(sample_rate: f64) -> Self {
        Self { chunks: Vec::new(), sample_rate }
    }
    pub fn push(&mut self, chunk: &'a [f32]) {
        self.chunks.push(chunk);
    }
    /// Produce a partial TrackAnalysis from whatever arrived so far
    /// (online fallback when offline pre-analysis is impossible).
    pub fn partial_analysis(&mut self) -> Option<TrackAnalysis> {
        let total: usize = self.chunks.iter().map(|c| c.len()).sum();
        if total < (self.sample_rate as usize) / 4 {
            return None;
        }
        let mut mono = Vec::with_capacity(total);
        for c in &self.chunks {
            mono.extend_from_slice(c);
        }
        let buf = AudioBuffer::new(self.sample_rate, 1, mono);
        let mut eng = crate::analysis::AnalysisEngine::default();
        Some(eng.analyze(&buf))
    }
}

/// `interface AudioAnalyzer`
pub trait AudioAnalyzer {
    fn analyze_track(&mut self, buffer: &AudioBuffer, meta: TrackMetadata) -> TrackAnalysis;
    fn analyze_stream(&mut self, stream: &mut AnalysisStream<'_>) -> Option<TrackAnalysis> {
        stream.partial_analysis()
    }
}

impl AudioAnalyzer for crate::analysis::AnalysisEngine {
    fn analyze_track(&mut self, buffer: &AudioBuffer, meta: TrackMetadata) -> TrackAnalysis {
        self.analyze_with_meta(buffer, meta)
    }
}

/// `interface TransitionPlanner`
pub trait TransitionPlannerApi {
    fn plan_transition(
        &self,
        source: &TrackAnalysis,
        target: &TrackAnalysis,
        context: &PlaybackContext,
        preference: &UserMixPreference,
    ) -> TransitionPlan;

    fn score_compatibility(&self, source: &TrackAnalysis, target: &TrackAnalysis) -> CompatibilityScore;
}

/// `interface MixEngine` — realtime contract. The host calls `process` from
/// its audio callback with interleaved output frames; sources are pulled from
/// registered players.
pub trait MixEngineApi {
    /// Arm a transition plan; scheduling becomes sample-accurate from now on.
    fn start(&mut self, plan: &TransitionPlan, samples_per_frame: usize, sample_rate: f64);
    /// Render one block of stereo output. Implementations must be
    /// allocation-free and RT-safe.
    fn process(&mut self, out: &mut [f32]);
    /// Manual override of an automation parameter (assist mode / user tweaks).
    fn set_automation(&mut self, param: MixParam, role: TrackRole, value: f64);
    fn stop(&mut self);
}

/// Controller modes for AutoMixController.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControllerMode {
    /// Fully automatic transitions.
    Auto,
    /// System proposes, user confirms (hints + armed cue points).
    Assist,
    /// Manual only.
    Off,
}

/// `interface AutoMixController`
pub trait AutoMixControllerApi {
    fn enable(&mut self, mode: ControllerMode);
    fn set_preference(&mut self, preference: UserMixPreference);
    fn next_track_hint(&self) -> Option<TrackHint>;
}
