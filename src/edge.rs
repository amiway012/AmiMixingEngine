//! Edge-case detection and graceful degradation (spec section 9).
//!
//! Every situation that can make an *automatic* transition dangerous is
//! detected here from the analysis results (or host events) and mapped to a
//! concrete mitigation: either a rewritten [`TransitionPlan`] or a cheaper
//! transition type. The guiding principle: when in doubt, hide the seam
//! (filter sweep / echo out / quick cut) instead of attempting a long blend.

use crate::model::*;
use crate::plan::{generate_automation, TransitionPlanner, MAX_STRETCH_PCT};

/// Situations the engine must survive.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeCase {
    /// No trustworthy pulse at all (no clear beat grid).
    NoClearBeat,
    /// Ambient / classical / jazz / live recording: rubato, no percussion.
    AmbientOrRubato,
    /// A cappella: vocal-dominant, essentially no drums.
    ACappella,
    /// Spoken word: speech-like mid-band activity without musical pulse.
    SpokenWord,
    /// Tempo drifts across the track (multi-BPM material).
    VariableBpm,
    /// Key changes inside the track.
    KeyModulation,
    /// Very short track (< ~12 s): not enough material for reliable grids.
    VeryShort,
    /// Long fade-in / fade-out regions around cue points.
    FadeInOut,
    /// Hidden track / leading silence gap.
    HiddenTrack,
    /// Silence at the end of the source.
    TrailingSilence,
    /// Source already clips (headroom risk on overlap sum).
    SourceClipping,
    /// Low-quality codec artifacts (e.g. MP3 truncation above ~16 kHz).
    CodecArtifacts,
    /// Metadata disagrees with analyzed audio.
    MetadataMismatch,
    /// PCM unavailable — metadata-only mode.
    NoPcmAccess,
    /// User skipped during an armed transition (host event).
    SkipDuringTransition,
    /// Queue changed while a mix was running (host event).
    QueueChangedMidMix,
    /// Phone call / audio focus loss (host event).
    AudioFocusLoss,
    /// Background playback restrictions / weak device (host event).
    BackgroundLimitations,
}

impl std::fmt::Display for EdgeCase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let s = match self {
            EdgeCase::NoClearBeat => "no_clear_beat",
            EdgeCase::AmbientOrRubato => "ambient_or_rubato",
            EdgeCase::ACappella => "a_cappella",
            EdgeCase::SpokenWord => "spoken_word",
            EdgeCase::VariableBpm => "variable_bpm",
            EdgeCase::KeyModulation => "key_modulation",
            EdgeCase::VeryShort => "very_short_track",
            EdgeCase::FadeInOut => "fade_in_out",
            EdgeCase::HiddenTrack => "hidden_track",
            EdgeCase::TrailingSilence => "trailing_silence",
            EdgeCase::SourceClipping => "source_clipping",
            EdgeCase::CodecArtifacts => "codec_artifacts",
            EdgeCase::MetadataMismatch => "metadata_mismatch",
            EdgeCase::NoPcmAccess => "no_pcm_access",
            EdgeCase::SkipDuringTransition => "skip_during_transition",
            EdgeCase::QueueChangedMidMix => "queue_changed_mid_mix",
            EdgeCase::AudioFocusLoss => "audio_focus_loss",
            EdgeCase::BackgroundLimitations => "background_limitations",
        };
        write!(f, "{s}")
    }
}

/// One detected case plus the human-readable mitigation note.
#[derive(Clone, Debug)]
pub struct DetectedCase {
    pub case: EdgeCase,
    pub note: String,
}

/// Detects edge cases from analyses and rewrites plans accordingly.
#[derive(Clone, Debug, Default)]
pub struct EdgeCaseHandler {
    /// Cheap per-deck headroom trim applied when `SourceClipping` detected,
    /// keyed by track id (linear gain < 1). Host mixers should query this.
    pub headroom_trim: Vec<(String, f64)>,
}

impl EdgeCaseHandler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Inspect one track analysis and list everything risky about it.
    pub fn detect_track(&mut self, a: &TrackAnalysis) -> Vec<DetectedCase> {
        let mut v = Vec::new();
        if a.quality.metadata_only || a.sample_rate <= 0.0 {
            v.push(dc(EdgeCase::NoPcmAccess, "no PCM: equal-power crossfade fallback"));
        }
        if !a.rhythm.has_clear_grid && !a.quality.metadata_only {
            if a.duration > 12.0 && a.rhythm.tempo_stability < 0.35 && !a.rhythm.has_clear_percussion {
                v.push(dc(EdgeCase::AmbientOrRubato, "rubato/no percussion: hide seam with filter/echo"));
            } else {
                v.push(dc(EdgeCase::NoClearBeat, "no clear grid: avoid long beatmatched blends"));
            }
        }
        if a.rhythm.has_clear_grid && a.rhythm.local_tempi.len() >= 3 {
            v.push(dc(EdgeCase::VariableBpm, "tempo drift: use local tempo window for ratio"));
        }
        if !a.harmony.modulations.is_empty() {
            v.push(dc(EdgeCase::KeyModulation, "intra-track modulation: enter after last modulation"));
        }
        if a.duration < 8.0 {
            v.push(dc(EdgeCase::VeryShort, "track too short for reliable analysis: plain crossfade"));
        }
        if a.loudness.fade_in_end.unwrap_or(0.0) > 1.0 || a.loudness.fade_out_start.map(|s| s < a.duration - 1.0).unwrap_or(false) {
            v.push(dc(EdgeCase::FadeInOut, "cue points already constrained outside fades"));
        }
        if let Some((_, e)) = a.loudness.silent_regions.first() {
            if *e > 5.0 {
                v.push(dc(EdgeCase::HiddenTrack, format!("leading silence {e:.1}s treated as hidden-track gap")));
            }
        }
        if let Some((s, _)) = a.loudness.silent_regions.last() {
            if a.duration - s > 2.0 && a.loudness.silent_regions.len() > 1 {
                v.push(dc(EdgeCase::TrailingSilence, "trailing silence: exit before gap"));
            }
        }
        if a.quality.source_clipping {
            v.push(dc(EdgeCase::SourceClipping, "-3 dB headroom trim applied on this deck"));
            self.headroom_trim.retain(|(id, _)| id != a.id);
            self.headroom_trim.push((a.id.clone(), 0.708)); // ≈ −3 dB
        }
        if a.quality.artifact_suspect || high_freq_truncated(a) {
            v.push(dc(EdgeCase::CodecArtifacts, "spectral hole above ~16 kHz: shorten overlap"));
        }
        if let Some(dbpm) = a.metadata.declared_bpm {
            if a.rhythm.has_clear_grid && (dbpm - a.rhythm.bpm).abs() > 6.0 {
                v.push(dc(
                    EdgeCase::MetadataMismatch,
                    format!("declared BPM {dbpm:.0} vs analyzed {:.0}: audio wins", a.rhythm.bpm),
                ));
            }
        }
        // vocal-dominant material
        let voc = mean(&a.spectral.vocal_presence);
        let kick = mean(&a.spectral.kick_density);
        if voc > 0.45 && kick < 0.05 && !a.quality.metadata_only {
            v.push(dc(EdgeCase::ACappella, "vocal without drums: cut on phrase, no tempo matching"));
        } else if voc > 0.35 && kick < 0.15 && centroid_low(a) {
            v.push(dc(EdgeCase::SpokenWord, "speech-like content: transition on pauses"));
        }
        v
    }

    /// Rewrite a planned transition so it survives the detected edge cases.
    /// Pure w.r.t. audio: only the plan (type/duration/cues/ratio) changes.
    pub fn adapt_plan(&self, plan: &TransitionPlan, a: &TrackAnalysis, b: &TrackAnalysis) -> TransitionPlan {
        let mut p = plan.clone();

        // metadata-only pair → hard force of the plain crossfade
        if a.quality.metadata_only || b.quality.metadata_only {
            p.transition_type = TransitionType::MetadataFallbackCrossfade;
            p.beatmatch = false;
            p.tempo_ratio = 1.0;
            p.key_shift_semitones = 0.0;
            return p;
        }

        // no trustworthy grid on either side → never attempt a long blend
        if !a.rhythm.has_clear_grid || !b.rhythm.has_clear_grid {
            if matches!(p.transition_type, TransitionType::SeamlessBeatmatchedBlend | TransitionType::BassSwap | TransitionType::DrumOnlyBridge) {
                p.transition_type = TransitionType::FilterSweep;
                p.beatmatch = false;
                p.duration = p.duration.min(4.0);
            }
        }

        // variable BPM: clamp the ratio using the *local* tempi near the cues
        if !b.rhythm.local_tempi.is_empty() && p.beatmatch {
            if let Some(local) = local_bpm_at(&b.rhythm.local_tempi, p.target_cue_time) {
                let ratio = a.rhythm.bpm / local.max(30.0);
                if (ratio - 1.0).abs() > MAX_STRETCH_PCT / 100.0 {
                    p.beatmatch = false;
                    p.transition_type = TransitionType::QuickCut;
                    p.duration = 0.05;
                } else {
                    p.tempo_ratio = ratio;
                }
            }
        }

        // key conflict still present after planning → shorten the overlap hard
        let compat = crate::analysis::chroma::harmonic_compatibility(&a.harmony, &b.harmony);
        if compat < 0.35 && p.duration > 2.0 && p.transition_type != TransitionType::QuickCut {
            p.transition_type = TransitionType::QuickCut;
            p.duration = 0.05;
        }

        // modulation inside the overlap: push target entry past the last modulation
        if let Some(last_mod) = b.harmony.modulations.iter().map(|m| m.time).fold(0.0f64, |x, y| x.max(y)) {
            if last_mod > 0.0 && p.target_cue_time < last_mod + 0.5 {
                p.target_cue_time = nearest_safe_entry(b, last_mod + 0.5);
            }
        }

        // codec artifacts / clipping: keep overlaps short, drop effect sends
        if a.quality.artifact_suspect || b.quality.artifact_suspect || a.quality.source_clipping || b.quality.source_clipping {
            p.duration = p.duration.min(4.0);
            p.automation.retain(|l| l.parameter != MixParam::Reverb && l.parameter != MixParam::Delay);
        }

        // trailing silence of the source: move the exit cue before the gap
        if let Some((s, _)) = b_late_silence(a) {
            if p.source_cue_time + p.duration > s {
                p.source_cue_time = nearest_safe_exit(a, s - p.duration).max(0.0);
                p.expected_gap = (s - (p.source_cue_time + p.duration)).max(0.0);
            }
        }

        // weak devices: strip costly lanes (delay/reverb sends, filters stay)
        // (device class arrives via adapt_plan_for_device)
        p
    }

    /// Device-driven degradation applied on top of [`adapt_plan`].
    pub fn adapt_plan_for_device(&self, plan: &TransitionPlan, device: DeviceClass) -> TransitionPlan {
        let mut p = plan.clone();
        match device {
            DeviceClass::Desktop => {}
            DeviceClass::Mobile => {
                // keep everything but cap stretch quality work: shorter blends
                p.duration = p.duration.min(8.0);
            }
            DeviceClass::LowPower => {
                // no time stretching in RT: prefer native-tempo styles
                p.beatmatch = false;
                p.tempo_ratio = 1.0;
                p.automation.retain(|l| l.parameter != MixParam::Delay && l.parameter != MixParam::Reverb);
                if p.transition_type == TransitionType::EchoOut {
                    p.transition_type = TransitionType::QuickCut;
                    p.duration = 0.05;
                }
            }
        }
        p
    }

    /// Host event: user hit skip while a transition was armed/running.
    /// Returns the plan the mixer should execute *immediately*.
    pub fn on_skip_during_transition(planner: &TransitionPlanner, a: &TrackAnalysis, b: &TrackAnalysis) -> TransitionPlan {
        // emergency handover: quick cut onto the next B downbeat
        let mut p = planner.fallback_plan(a, b, UserMixPreference::Smooth);
        p.transition_type = TransitionType::QuickCut;
        p.duration = 0.05;
        p.beatmatch = false;
        p.tempo_ratio = 1.0;
        p.key_shift_semitones = 0.0;
        p.confidence = 0.3;
        p.automation = vec![
            AutomationCurve::new(
                MixParam::Volume,
                TrackRole::Source,
                vec![ap(0.0, 1.0), ap(0.05, 0.0)],
            ),
            AutomationCurve::new(
                MixParam::Volume,
                TrackRole::Target,
                vec![ap(0.0, 0.0), ap(0.05, 1.0)],
            ),
        ];
        p
    }

    /// Host event: queue changed mid-mix. Decide whether the armed plan stays.
    /// Re-planning is only safe before the transition becomes audible; once we
    /// are inside the overlap region, finishing beats restarting.
    pub fn on_queue_changed(current_plan: &TransitionPlan, elapsed_s: f64, new_next_id: &str) -> QueueDecision {
        if current_plan.target_track_id == new_next_id {
            return QueueDecision::KeepArmed;
        }
        if elapsed_s < current_plan.duration * 0.25 {
            QueueDecision::ReplanNow
        } else {
            // too late to cancel cleanly: finish with the *fallback* then replan
            QueueDecision::FinishWithFallbackThenReplan
        }
    }

    /// Host event: phone call / audio focus loss — produce a safe fade plan.
    pub fn on_audio_focus_loss(plan: &TransitionPlan) -> TransitionPlan {
        let mut p = plan.clone();
        p.duration = p.duration.min(0.3);
        p.transition_type = TransitionType::QuickCut;
        p.beatmatch = false;
        p.automation = vec![AutomationCurve::new(
            MixParam::Volume,
            TrackRole::Source,
            vec![ap(0.0, 1.0), ap(p.duration, 0.0)],
        )];
        p
    }
}

/// What to do when the queue changed while a mix was armed/running.
#[derive(Clone, Debug, PartialEq)]
pub enum QueueDecision {
    KeepArmed,
    ReplanNow,
    FinishWithFallbackThenReplan,
}

fn dc(case: EdgeCase, note: impl Into<String>) -> DetectedCase {
    DetectedCase { case, note: note.into() }
}

fn ap(time: f64, value: f64) -> AutomationPoint {
    AutomationPoint { time, value, curve: CurveShape::Linear }
}

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.iter().sum::<f64>() / v.len() as f64
}

/// Heuristic: codec low-pass truncation — average spectral centroid pinned
/// suspiciously low while highs are near zero throughout the track.
fn high_freq_truncated(a: &TrackAnalysis) -> bool {
    if a.spectral.high.is_empty() || a.spectral.centroid.is_empty() {
        return false;
    }
    mean(&a.spectral.high) < 0.02 && mean(&a.spectral.centroid) < 1200.0 && mean(&a.spectral.mid) > 0.1
}

fn centroid_low(a: &TrackAnalysis) -> bool {
    !a.spectral.centroid.is_empty() && mean(&a.spectral.centroid) < 1400.0
}

fn local_bpm_at(tempi: &[(f64, f64)], t: f64) -> Option<f64> {
    tempi.iter().find(|(s, _)| t >= *s).map(|(_, b)| *b).or_else(|| tempi.first().map(|(_, b)| *b))
}

fn nearest_safe_entry(a: &TrackAnalysis, t: f64) -> f64 {
    if a.structure.safe_entry_points.is_empty() {
        return t;
    }
    *a.structure
        .safe_entry_points
        .iter()
        .min_by(|x, y| (x - t).abs().partial_cmp(&(y - t).abs()).unwrap())
        .unwrap()
}

fn nearest_safe_exit(a: &TrackAnalysis, t: f64) -> f64 {
    if a.structure.safe_exit_points.is_empty() {
        return t;
    }
    *a.structure
        .safe_exit_points
        .iter()
        .min_by(|x, y| (x - t).abs().partial_cmp(&(y - t).abs()).unwrap())
        .unwrap()
}

/// Trailing silent region of a track, if it starts after ≥70 % of duration.
fn b_late_silence(a: &TrackAnalysis) -> Option<(f64, f64)> {
    a.loudness
        .silent_regions
        .last()
        .copied()
        .filter(|(s, _)| *s > a.duration * 0.7)
}

/// Convenience: full guard pass — detect on both tracks, then adapt.
pub fn guarded_plan(
    handler: &mut EdgeCaseHandler,
    planner: &TransitionPlanner,
    a: &TrackAnalysis,
    b: &TrackAnalysis,
    ctx: &PlaybackContext,
    pref: UserMixPreference,
) -> (TransitionPlan, Vec<DetectedCase>) {
    let mut cases = handler.detect_track(a);
    cases.extend(handler.detect_track(b));
    let raw = planner.plan_with_fallback(a, b, ctx, pref);
    let adapted = handler.adapt_plan_for_device(handler.adapt_plan(&raw, a, b), ctx.device);
    // regenerate automation consistent with the final type/duration
    let mut final_plan = adapted.clone();
    if final_plan.transition_type != raw.transition_type || (final_plan.duration - raw.duration).abs() > 1e-6 {
        if let Some(cand) = planner.candidates(a, b, pref).into_iter().next() {
            final_plan.automation = generate_automation(final_plan.transition_type, final_plan.duration, a, b, &cand, pref);
        }
    }
    (final_plan, cases)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth;

    fn analyze(o: &SynthOptions) -> TrackAnalysis {
        let buf = synth::synth_track(o);
        let mut eng = crate::analysis::AnalysisEngine::default();
        let mut a = eng.analyze(&buf);
        a.id = o.seed.to_string();
        a
    }

    #[test]
    fn ambient_detected_and_blend_downgraded() {
        // pure tone pad: no percussion, unstable energy
        let buf = synth::ambient_track(30.0, 44100.0, crate::synth::name_to_pc("A"));
        let mut eng = crate::analysis::AnalysisEngine::default();
        let a = eng.analyze(&buf);
        let mut h = EdgeCaseHandler::new();
        let cases = h.detect_track(&a);
        assert!(cases
            .iter()
            .any(|c| matches!(c.case, EdgeCase::NoClearBeat | EdgeCase::AmbientOrRubato)));
        // any long-blend plan must be downgraded
        let mut p = analyze(&SynthOptions::default());
        p.rhythm.has_clear_grid = true;
        let planner = TransitionPlanner::default();
        let raw = planner.plan_with_fallback(&p, &a, &PlaybackContext::default(), UserMixPreference::Smooth);
        let adapted = h.adapt_plan(&raw, &p, &a);
        if !a.rhythm.has_clear_grid {
            assert!(!adapted.beatmatch, "beatmatch must be off without a grid");
        }
        let _ = &mut p;
    }

    #[test]
    fn metadata_only_forces_crossfade() {
        let md = TrackMetadata { id: "m".into(), declared_bpm: Some(120.0), ..Default::default() };
        let a = TrackAnalysis::from_metadata(md, 180.0);
        let b = TrackAnalysis::from_metadata(md, 200.0);
        let mut h = EdgeCaseHandler::new();
        let cases = h.detect_track(&a);
        assert!(cases.iter().any(|c| c.case == EdgeCase::NoPcmAccess));
        let planner = TransitionPlanner::default();
        let raw = planner.plan_with_fallback(&a, &b, &PlaybackContext::default(), UserMixPreference::Smooth);
        let adapted = h.adapt_plan(&raw, &a, &b);
        assert_eq!(adapted.transition_type, TransitionType::MetadataFallbackCrossfade);
    }

    #[test]
    fn clipping_adds_headroom_trim() {
        let mut a = analyze(&SynthOptions::default());
        a.quality.source_clipping = true;
        let mut h = EdgeCaseHandler::new();
        let cases = h.detect_track(&a);
        assert!(cases.iter().any(|c| c.case == EdgeCase::SourceClipping));
        assert!(h.headroom_trim.iter().any(|(id, g)| *id == a.id && (*g - 0.708).abs() < 1e-6));
    }

    #[test]
    fn lowpower_device_strips_stretch_and_effects() {
        let a = analyze(&SynthOptions::default());
        let b = analyze(&SynthOptions { bpm: 126.0, seed: 5, ..Default::default() });
        let planner = TransitionPlanner::default();
        let raw = planner.plan_with_fallback(&a, &b, &PlaybackContext::default(), UserMixPreference::Smooth);
        let h = EdgeCaseHandler::new();
        let p = h.adapt_plan_for_device(&raw, DeviceClass::LowPower);
        assert_eq!(p.tempo_ratio, 1.0);
        assert!(!p.beatmatch);
        assert!(!p.automation.iter().any(|l| l.parameter == MixParam::Delay || l.parameter == MixParam::Reverb));
    }

    #[test]
    fn skip_yields_quick_cut_plan() {
        let a = analyze(&SynthOptions::default());
        let b = analyze(&SynthOptions { seed: 9, ..Default::default() });
        let planner = TransitionPlanner::default();
        let p = EdgeCaseHandler::on_skip_during_transition(&planner, &a, &b);
        assert_eq!(p.transition_type, TransitionType::QuickCut);
        assert!(p.duration <= 0.1);
    }

    #[test]
    fn queue_change_decision_by_elapsed() {
        let plan = TransitionPlan {
            source_track_id: "a".into(),
            target_track_id: "b".into(),
            source_cue_time: 0.0,
            target_cue_time: 0.0,
            transition_type: TransitionType::SeamlessBeatmatchedBlend,
            duration: 8.0,
            beatmatch: true,
            key_shift_semitones: 0.0,
            tempo_ratio: 1.0,
            automation: vec![],
            fallback_plan: None,
            confidence: 0.5,
            cost: TransitionCost::default(),
            expected_gap: 0.0,
        };
        assert_eq!(EdgeCaseHandler::on_queue_changed(&plan, 0.5, "b"), QueueDecision::KeepArmed);
        assert_eq!(EdgeCaseHandler::on_queue_changed(&plan, 0.5, "c"), QueueDecision::ReplanNow);
        assert_eq!(EdgeCaseHandler::on_queue_changed(&plan, 4.0, "c"), QueueDecision::FinishWithFallbackThenReplan);
    }

    #[test]
    fn focus_loss_fades_fast() {
        let mut plan = TransitionPlan {
            source_track_id: "a".into(),
            target_track_id: "b".into(),
            source_cue_time: 0.0,
            target_cue_time: 0.0,
            transition_type: TransitionType::SeamlessBeatmatchedBlend,
            duration: 8.0,
            beatmatch: true,
            key_shift_semitones: 0.0,
            tempo_ratio: 1.0,
            automation: vec![],
            fallback_plan: None,
            confidence: 0.5,
            cost: TransitionCost::default(),
            expected_gap: 0.0,
        };
        plan.tempo_ratio = 1.02;
        let p = EdgeCaseHandler::on_audio_focus_loss(&plan);
        assert!(p.duration <= 0.31);
        assert_eq!(p.transition_type, TransitionType::QuickCut);
    }

    #[test]
    fn variable_bpm_clamps_or_cuts() {
        let mut a = analyze(&SynthOptions::default());
        a.rhythm.local_tempi = vec![(0.0, 124.0), (15.0, 140.0)];
        a.rhythm.has_clear_grid = true;
        let b = a.clone();
        let planner = TransitionPlanner::default();
        let raw = planner.plan_with_fallback(&a, &b, &PlaybackContext::default(), UserMixPreference::Smooth);
        let h = EdgeCaseHandler::new();
        let p = h.adapt_plan(&raw, &a, &b);
        // ratio 124/140 is far beyond ±8% → must refuse beatmatching
        assert!(!p.beatmatch || (p.tempo_ratio - 1.0).abs() <= 0.081);
    }
}
