//! Transition planning (spec sections 2, 3).
//!
//! The planner enumerates candidate (exit-point × entry-point) pairs from the
//! structural safe points of both tracks, phase-aligns them on the beat grid,
//! scores every candidate with the weighted cost function and finally picks a
//! transition *type* according to compatibility rules. Every plan carries a
//! fallback plan so the mixer can degrade gracefully mid-transition.

use crate::analysis::chroma::harmonic_compatibility;
use crate::api::TransitionPlannerApi;
use crate::dsp::stretch::stretch_artifact_risk;
use crate::model::*;

/// Maximum tolerated tempo mismatch for time-stretching (±8 %).
pub const MAX_STRETCH_PCT: f64 = 8.0;
/// Preferred stretch range (±6 %) — beyond it artifact penalty grows fast.
pub const SAFE_STRETCH_PCT: f64 = 6.0;
/// Beat-alignment error that saturates the penalty (ms).
const BEAT_SATURATION_MS: f64 = 50.0;
/// Loudness jump (LU) that saturates the energy-jump penalty.
const ENERGY_JUMP_SATURATION_LU: f64 = 6.0;
/// Key confidence below which a pitch shift is never proposed.
const KEY_SHIFT_MIN_CONFIDENCE: f64 = 0.7;

/// The concrete transition planner.
#[derive(Clone, Debug)]
pub struct TransitionPlanner {
    pub weights: CostWeights,
    /// Allow ±1–2 semitone key shifts (off by default per spec §4).
    pub enable_key_shift: bool,
    /// Max number of candidates evaluated per pair.
    pub max_candidates: usize,
}

impl Default for TransitionPlanner {
    fn default() -> Self {
        Self { weights: CostWeights::default(), enable_key_shift: false, max_candidates: 64 }
    }
}

/// One enumerated candidate (exit-point × entry-point pair) before typing.
/// Exposed publicly so hosts can inspect the ranked candidate list and so
/// `edge`/`mix` can regenerate automation for an adapted plan.
#[derive(Clone, Copy, Debug)]
pub struct Candidate {
    /// time inside track A where the transition starts (source cue), s
    pub exit_t: f64,
    /// time inside track B where playback begins (target cue), s
    pub entry_t: f64,
    /// planned overlap length in bars of A
    pub overlap_bars: f64,
    /// full weighted cost breakdown
    pub cost: TransitionCost,
    /// downbeat alignment error in ms after phase search
    pub align_ms: f64,
}

// ---------------------------------------------------------------------------
// Feature extraction helpers
// ---------------------------------------------------------------------------

/// Short-term LUFS at time `t` (falls back to integrated loudness).
fn short_term_lufs(a: &TrackAnalysis, t: f64) -> f64 {
    if a.loudness.short_term.is_empty() {
        return a.loudness.integrated_lufs;
    }
    let hop = a.duration / a.loudness.short_term.len() as f64;
    let i = ((t / hop.max(1e-9)) as usize).min(a.loudness.short_term.len() - 1);
    a.loudness.short_term[i]
}

/// Mean product of low-band energies of A and B across an overlap region.
fn bass_conflict(a: &TrackAnalysis, b: &TrackAnalysis, start_a: f64, start_b: f64, dur: f64) -> f64 {
    if a.spectral.low.is_empty() || b.spectral.low.is_empty() || dur <= 0.0 {
        return 0.0;
    }
    let hop = a.spectral.hop.max(b.spectral.hop).max(0.05);
    let n = (dur / hop).max(1.0) as usize;
    let mut acc = 0.0;
    for k in 0..n {
        let d = k as f64 * hop;
        let ia = a.spectral.idx_at(start_a + d);
        let ib = b.spectral.idx_at(start_b + d);
        acc += a.spectral.low[ia] * b.spectral.low[ib];
    }
    (acc / n as f64).clamp(0.0, 1.0)
}

/// Vocal collision: excess simultaneous vocal presence over the overlap.
fn vocal_collision(a: &TrackAnalysis, b: &TrackAnalysis, start_a: f64, start_b: f64, dur: f64) -> f64 {
    if a.spectral.vocal_presence.is_empty() || b.spectral.vocal_presence.is_empty() || dur <= 0.0 {
        return 0.0;
    }
    let hop = a.spectral.hop.max(b.spectral.hop).max(0.05);
    let n = (dur / hop).max(1.0) as usize;
    let mut acc = 0.0;
    for k in 0..n {
        let d = k as f64 * hop;
        let va = a.spectral.vocal_presence[a.spectral.idx_at(start_a + d)];
        let vb = b.spectral.vocal_presence[b.spectral.idx_at(start_b + d)];
        acc += (va + vb - 0.7).max(0.0);
    }
    (acc / n as f64 / 0.6).clamp(0.0, 1.0)
}

/// Structural mismatch table (spec §3): exiting section → entering section.
fn structural_mismatch(exit_label: SegmentLabel, entry_label: SegmentLabel) -> f64 {
    use SegmentLabel::*;
    match (exit_label, entry_label) {
        // dropping straight into a breakdown kills the dancefloor
        (Drop, Breakdown) | (Chorus, Breakdown) => 1.0,
        // breakdown → drop is *prepared* energy release: cheap
        (Breakdown, Drop) | (BuildUp, Drop) => 0.15,
        (Drop, Verse) | (Chorus, Verse) => 0.6,
        (Verse, Chorus) | (Verse, Drop) => 0.3,
        (Intro, _) | (_, Outro) | (_, Tail) | (SilentGap, _) | (_, SilentGap) => 0.5,
        (Outro, Intro) => 0.1,
        _ => 0.35,
    }
}

fn label_at(structure: &TrackStructure, t: f64) -> SegmentLabel {
    structure
        .segments
        .iter()
        .find(|s| t >= s.start && t < s.end)
        .map(|s| s.label)
        .unwrap_or(SegmentLabel::Unknown)
}

/// Downbeat alignment error (ms) between A exiting at `exit_t` and B entering
/// at `entry_t`, searching one bar of relative phase offsets (grid slip).
fn beat_alignment_ms(a: &TrackAnalysis, b: &TrackAnalysis, exit_t: f64, entry_t: f64) -> f64 {
    if !a.rhythm.has_clear_grid || !b.rhythm.has_clear_grid {
        return BEAT_SATURATION_MS;
    }
    let spba = 60.0 / a.rhythm.bpm.max(1.0);
    let spbb = 60.0 / b.rhythm.bpm.max(1.0);
    let mut best = f64::INFINITY;
    // candidate integer-beat slips around the naive pairing
    for k in -8..=8i64 {
        // A beat position after slipping k beats of A's grid
        let shifted = exit_t + k as f64 * spba;
        // target time on B's timeline that should coincide with it
        let target = entry_t + (shifted - exit_t) / spba * spbb;
        let db = nearest_time(&b.rhythm.downbeat_times, target);
        let err = (db - target).abs();
        // also account for accumulated drift over the lookahead horizon:
        // beats stay locked only while ratio error is small; approximate the
        // worst-case drift over 16 bars.
        let ratio_err = (spbb / spba - 1.0).abs();
        let drift = ratio_err * spbb * b.rhythm.meter.max(1) as f64 * 16.0;
        let total = (err.min((spbb - err).abs())).max(drift.min(spbb / 2.0));
        best = best.min(total);
    }
    (best * 1000.0).min(BEAT_SATURATION_MS)
}

fn nearest_time(times: &[f64], t: f64) -> f64 {
    if times.is_empty() {
        return t;
    }
    let mut lo = 0usize;
    let mut hi = times.len() - 1;
    while lo + 1 < hi {
        let mid = (lo + hi) / 2;
        if times[mid] <= t {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    if (times[lo] - t).abs() <= (times[hi] - t).abs() {
        times[lo]
    } else {
        times[hi]
    }
}

/// Phrase misalignment: distance of the cue point to the nearest 4-bar phrase
/// boundary, normalized by half a phrase (0 = exactly on the phrase).
fn phrase_misalignment(a: &TrackAnalysis, t: f64) -> f64 {
    if a.rhythm.bpm <= 0.0 {
        return 1.0;
    }
    let phrase = (60.0 / a.rhythm.bpm) * a.rhythm.meter.max(1) as f64 * 4.0;
    let anchor = a.structure.phrase_boundaries.first().copied().unwrap_or(0.0);
    let d = ((t - anchor) % phrase).abs();
    let d = d.min(phrase - d);
    (d / (phrase / 2.0)).clamp(0.0, 1.0)
}

// ---------------------------------------------------------------------------
// Planner implementation
// ---------------------------------------------------------------------------

impl TransitionPlanner {
    /// Compatibility summary used for queue hints and quick decisions.
    pub fn compatibility(&self, a: &TrackAnalysis, b: &TrackAnalysis) -> CompatibilityScore {
        let bpm_delta_abs = (a.rhythm.bpm - b.rhythm.bpm).abs();
        let bpm_delta_pct = if a.rhythm.bpm > 0.0 {
            bpm_delta_abs / a.rhythm.bpm * 100.0
        } else {
            100.0
        };
        let harmonic = harmonic_compatibility(&a.harmony, &b.harmony);
        let rhythmic = if a.rhythm.has_clear_grid && b.rhythm.has_clear_grid {
            (1.0 - (bpm_delta_pct / MAX_STRETCH_PCT).min(1.0))
                * a.rhythm.bpm_confidence.max(0.2)
                * b.rhythm.bpm_confidence.max(0.2)
        } else {
            0.15
        };
        let energy = 1.0 - ((a.loudness.integrated_lufs - b.loudness.integrated_lufs).abs() / 9.0).min(1.0);
        let struct_score = {
            let ea = a.structure.segments.last().map(|s| s.energy).unwrap_or(0.5);
            let eb = b.structure.segments.first().map(|s| s.energy).unwrap_or(0.5);
            1.0 - (ea - eb).abs().clamp(0.0, 1.0)
        };
        let overall = (harmonic * 0.35 + rhythmic * 0.35 + energy * 0.15 + struct_score * 0.15)
            .clamp(0.0, 1.0);
        CompatibilityScore {
            overall,
            harmonic,
            rhythmic,
            energy,
            structural: struct_score,
            camelot_distance: camelot_distance(&a.harmony.camelot, &b.harmony.camelot),
            bpm_delta_abs,
            bpm_delta_pct,
        }
    }

    /// Enumerate and score all candidate transitions between two analyses.
    pub fn candidates(&self, a: &TrackAnalysis, b: &TrackAnalysis, preference: UserMixPreference) -> Vec<Candidate> {
        let w = self.weights.for_preference(preference);
        let compat = self.compatibility(a, b);
        let mut out = Vec::new();
        for &exit_t in &a.structure.safe_exit_points {
            for &entry_t in &b.structure.safe_entry_points {
                // overlap length: prefer 4–16 bars when grids exist
                let bar_b = if b.rhythm.bpm > 0.0 {
                    (60.0 / b.rhythm.bpm) * b.rhythm.meter.max(1) as f64
                } else {
                    4.0
                };
                let avail_a = (a.duration - exit_t).max(0.0);
                let overlap_bars = (8.0f64).min((avail_a / bar_b).max(1.0));
                let dur = overlap_bars * bar_b;

                let align_ms = beat_alignment_ms(a, b, exit_t, entry_t);
                let key_pen = 1.0
                    - harmonic_compatibility(
                        &key_at(&a.harmony, exit_t),
                        &key_at(&b.harmony, entry_t),
                    );
                let phrase_pen = (phrase_misalignment(a, exit_t) + phrase_misalignment(b, entry_t)) * 0.5;
                let bass_pen = bass_conflict(a, b, exit_t.max(0.0) , entry_t, dur);
                let vocal_pen = vocal_collision(a, b, exit_t, entry_t, dur);
                let jump = (short_term_lufs(a, exit_t) - short_term_lufs(b, entry_t)).abs();
                let energy_pen = (jump / ENERGY_JUMP_SATURATION_LU).min(1.0);
                let struct_pen = structural_mismatch(label_at(&a.structure, exit_t), label_at(&b.structure, entry_t));
                let ratio = if b.rhythm.bpm > 0.0 && a.rhythm.bpm > 0.0 {
                    a.rhythm.bpm / b.rhythm.bpm
                } else {
                    1.0
                };
                let mut art_pen = stretch_artifact_risk(ratio);
                if a.quality.source_clipping || b.quality.source_clipping {
                    art_pen = art_pen.max(0.4);
                }
                if a.quality.artifact_suspect || b.quality.artifact_suspect {
                    art_pen = (art_pen + 0.3).min(1.0);
                }
                // gap risk: silence between end-of-A and B entry would be audible
                let smooth = (overlap_bars / 8.0)
                    .min(1.0)
                    * a.rhythm.bpm_confidence
                    * b.rhythm.bpm_confidence
                    * if a.rhythm.has_clear_grid && b.rhythm.has_clear_grid { 1.0 } else { 0.3 };
                let interest = {
                    let camel_step = compat.camelot_distance == 1 || compat.camelot_distance == 7;
                    let drop_entry = matches!(label_at(&b.structure, entry_t), SegmentLabel::Drop);
                    (camel_step as u8 as f64 * 0.5 + drop_entry as u8 as f64 * 0.5).min(1.0)
                };

                let mut cost = TransitionCost {
                    beat_alignment_error: align_ms / BEAT_SATURATION_MS,
                    key_conflict_penalty: key_pen,
                    phrase_misalignment_penalty: phrase_pen,
                    bass_conflict_penalty: bass_pen,
                    vocal_collision_penalty: vocal_pen,
                    energy_jump_penalty: energy_pen,
                    structural_mismatch_penalty: struct_pen,
                    artifact_penalty: art_pen,
                    smoothness_bonus: smooth,
                    musical_interest_bonus: interest,
                    total: 0.0,
                };
                w.evaluate(&mut cost);
                out.push(Candidate { exit_t, entry_t, overlap_bars, cost, align_ms });
            }
        }
        out.sort_by(|x, y| x.cost.total.partial_cmp(&y.cost.total).unwrap_or(std::cmp::Ordering::Equal));
        out.truncate(self.max_candidates);
        out
    }

    /// Pick the transition type given the winning candidate and context.
    fn choose_type(&self, a: &TrackAnalysis, b: &TrackAnalysis, c: &Candidate, pref: UserMixPreference) -> TransitionType {
        if a.quality.metadata_only || b.quality.metadata_only {
            return TransitionType::MetadataFallbackCrossfade;
        }
        let grids_ok = a.rhythm.has_clear_grid && b.rhythm.has_clear_grid;
        let key_compat = harmonic_compatibility(&a.harmony, &b.harmony);
        let ratio_pct = if b.rhythm.bpm > 0.0 && a.rhythm.bpm > 0.0 {
            ((a.rhythm.bpm / b.rhythm.bpm) - 1.0).abs() * 100.0
        } else {
            100.0
        };
        if !grids_ok || b.rhythm.bpm_confidence < 0.15 {
            // no trustworthy pulse → hide the seam
            return match pref {
                UserMixPreference::Dramatic | UserMixPreference::Energetic => TransitionType::EchoOut,
                _ => TransitionType::FilterSweep,
            };
        }
        if ratio_pct > MAX_STRETCH_PCT || key_compat < 0.35 {
            return TransitionType::QuickCut;
        }
        let heavy_bass = c.cost.bass_conflict_penalty > 0.35;
        let vocal_clash = c.cost.vocal_collision_penalty > 0.4;
        if vocal_clash && key_compat >= 0.5 {
            return TransitionType::DrumOnlyBridge;
        }
        if heavy_bass && key_compat >= 0.55 {
            return TransitionType::BassSwap;
        }
        match pref {
            UserMixPreference::Energetic if matches!(label_at(&b.structure, c.entry_t), SegmentLabel::Drop) => {
                TransitionType::FilterSweep
            }
            UserMixPreference::Dramatic if ratio_pct <= SAFE_STRETCH_PCT && key_compat >= 0.5 => {
                TransitionType::EchoOut
            }
            _ => TransitionType::SeamlessBeatmatchedBlend,
        }
    }

    /// Suggest a key shift (semitones) if enabled and worthwhile.
    fn propose_key_shift(&self, a: &TrackAnalysis, b: &TrackAnalysis, compat: f64) -> f64 {
        if !self.enable_key_shift || compat >= 0.55 {
            return 0.0;
        }
        if a.harmony.confidence < KEY_SHIFT_MIN_CONFIDENCE || b.harmony.confidence < KEY_SHIFT_MIN_CONFIDENCE {
            return 0.0;
        }
        // try ±1, ±2 semitones on the Camelot wheel; pick the best improvement
        let names = NOTE_NAMES;
        let pc = names.iter().position(|n| *n == b.harmony.key).unwrap_or(0) as i32;
        let mut best = (0.0f64, compat);
        for st in [-2i32, -1, 1, 2] {
            let np = (pc + st + 12) % 12;
            let shifted = TrackHarmony {
                key: pc_to_name(np as usize).into(),
                scale: b.harmony.scale,
                camelot: camelot_for(np as usize, b.harmony.scale),
                chroma: rotate_chroma(&b.harmony.chroma, st),
                ..b.harmony.clone()
            };
            let v = harmonic_compatibility(&a.harmony, &shifted);
            if v > best.1 {
                best = (st as f64, v);
            }
        }
        if best.1 - compat > 0.15 {
            best.0
        } else {
            0.0
        }
    }

    /// Build the full plan (public API path).
    pub fn plan_with_fallback(
        &self,
        a: &TrackAnalysis,
        b: &TrackAnalysis,
        ctx: &PlaybackContext,
        preference: UserMixPreference,
    ) -> TransitionPlan {
        let cands = self.candidates(a, b, preference);
        let best = cands.into_iter().next();
        match best {
            Some(c) => self.materialize(a, b, &c, preference, ctx.force_now),
            None => self.fallback_plan(a, b, preference),
        }
    }

    fn materialize(
        &self,
        a: &TrackAnalysis,
        b: &TrackAnalysis,
        c: &Candidate,
        preference: UserMixPreference,
        force_now: bool,
    ) -> TransitionPlan {
        let ty = self.choose_type(a, b, c, preference);
        let bar_b = if b.rhythm.bpm > 0.0 {
            (60.0 / b.rhythm.bpm) * b.rhythm.meter.max(1) as f64
        } else {
            4.0
        };
        let duration = match ty {
            TransitionType::QuickCut => 0.05,
            TransitionType::MetadataFallbackCrossfade => (bar_b * 2.0).max(2.0).min(8.0),
            TransitionType::EchoOut => bar_b * 2.0,
            TransitionType::FilterSweep => bar_b * 4.0,
            TransitionType::DrumOnlyBridge => bar_b * 8.0,
            _ => bar_b * c.overlap_bars,
        };
        let duration = if force_now { duration.min(bar_b) } else { duration };
        let tempo_ratio = if a.rhythm.has_clear_grid && b.rhythm.has_clear_grid && b.rhythm.bpm > 0.0 {
            (a.rhythm.bpm / b.rhythm.bpm).clamp(1.0 - MAX_STRETCH_PCT / 100.0, 1.0 + MAX_STRETCH_PCT / 100.0)
        } else {
            1.0
        };
        let compat = harmonic_compatibility(&a.harmony, &b.harmony);
        let key_shift = self.propose_key_shift(a, b, compat);
        let automation = generate_automation(ty, duration, a, b, c, preference);
        let confidence = (1.0 - (c.cost.total / 6.0).clamp(0.0, 1.0)).clamp(0.05, 0.98);

        // fallback chain: EchoOut → QuickCut
        let fallback_ty = match ty {
            TransitionType::SeamlessBeatmatchedBlend | TransitionType::BassSwap | TransitionType::DrumOnlyBridge => {
                TransitionType::EchoOut
            }
            TransitionType::EchoOut | TransitionType::FilterSweep => TransitionType::QuickCut,
            _ => TransitionType::MetadataFallbackCrossfade,
        };
        let fallback = Box::new(TransitionPlan {
            source_track_id: a.id.clone(),
            target_track_id: b.id.clone(),
            source_cue_time: c.exit_t,
            target_cue_time: c.entry_t,
            transition_type: fallback_ty,
            duration: match fallback_ty {
                TransitionType::QuickCut => 0.05,
                TransitionType::EchoOut => duration * 0.5,
                _ => duration.min(bar_b * 2.0),
            },
            beatmatch: fallback_ty != TransitionType::QuickCut && a.rhythm.has_clear_grid,
            key_shift_semitones: 0.0,
            tempo_ratio,
            automation: generate_automation(fallback_ty, duration * 0.5, a, b, c, UserMixPreference::Smooth),
            fallback_plan: None,
            confidence: confidence * 0.7,
            cost: TransitionCost { total: c.cost.total + 0.5, ..c.cost },
            expected_gap: 0.0,
        });

        TransitionPlan {
            source_track_id: a.id.clone(),
            target_track_id: b.id.clone(),
            source_cue_time: c.exit_t,
            target_cue_time: c.entry_t,
            transition_type: ty,
            duration,
            beatmatch: matches!(
                ty,
                TransitionType::SeamlessBeatmatchedBlend
                    | TransitionType::BassSwap
                    | TransitionType::DrumOnlyBridge
                    | TransitionType::FilterSweep
            ) && a.rhythm.has_clear_grid
                && b.rhythm.has_clear_grid,
            key_shift_semitones: key_shift,
            tempo_ratio,
            automation,
            fallback_plan: Some(fallback),
            confidence,
            cost: c.cost,
            expected_gap: 0.0,
        }
    }

    /// Last-resort plan when nothing was analyzable.
    pub fn fallback_plan(&self, a: &TrackAnalysis, b: &TrackAnalysis, _pref: UserMixPreference) -> TransitionPlan {
        let dur = 4.0;
        TransitionPlan {
            source_track_id: a.id.clone(),
            target_track_id: b.id.clone(),
            source_cue_time: a.duration - dur,
            target_cue_time: 0.0,
            transition_type: TransitionType::MetadataFallbackCrossfade,
            duration: dur,
            beatmatch: false,
            key_shift_semitones: 0.0,
            tempo_ratio: 1.0,
            automation: vec![
                AutomationCurve::new(
                    MixParam::Volume,
                    TrackRole::Source,
                    vec![
                        AutomationPoint { time: 0.0, value: 1.0, curve: CurveShape::Linear },
                        AutomationPoint { time: dur, value: 0.0, curve: CurveShape::EqualPower },
                    ],
                ),
                AutomationCurve::new(
                    MixParam::Volume,
                    TrackRole::Target,
                    vec![
                        AutomationPoint { time: 0.0, value: 0.0, curve: CurveShape::Linear },
                        AutomationPoint { time: dur, value: 1.0, curve: CurveShape::EqualPower },
                    ],
                ),
            ],
            fallback_plan: None,
            confidence: 0.15,
            cost: TransitionCost::default(),
            expected_gap: 0.0,
        }
    }

    /// Rank a queue of candidates behind the currently playing track.
    pub fn rank_queue(&self, current: &TrackAnalysis, queue: &[&TrackAnalysis], preference: UserMixPreference) -> Vec<TrackHint> {
        let mut hints: Vec<TrackHint> = queue
            .iter()
            .map(|b| {
                let cs = self.compatibility(current, b);
                let cands = self.candidates(current, b, preference);
                let best = cands.first();
                let (ty, reason) = match best {
                    Some(c) => {
                        let t = self.choose_type(current, b, c, preference);
                        (
                            t,
                            format!(
                                "compat {:.0}% · camelot Δ{} · Δbpm {:.1}% · cost {:.2}",
                                cs.overall * 100.0,
                                cs.camelot_distance,
                                cs.bpm_delta_pct,
                                c.cost.total
                            ),
                        )
                    }
                    None => (
                        TransitionType::MetadataFallbackCrossfade,
                        format!("compat {:.0}% (no analyzable cues)", cs.overall * 100.0),
                    ),
                };
                TrackHint { track_id: b.id.clone(), suggested_transition: ty, score: cs.overall, reason }
            })
            .collect();
        hints.sort_by(|x, y| y.score.partial_cmp(&x.score).unwrap_or(std::cmp::Ordering::Equal));
        hints
    }
}

impl TransitionPlannerApi for TransitionPlanner {
    fn plan_transition(
        &self,
        source: &TrackAnalysis,
        target: &TrackAnalysis,
        context: &PlaybackContext,
        preference: &UserMixPreference,
    ) -> TransitionPlan {
        self.plan_with_fallback(source, target, context, *preference)
    }

    fn score_compatibility(&self, source: &TrackAnalysis, target: &TrackAnalysis) -> CompatibilityScore {
        self.compatibility(source, target)
    }
}

// ---------------------------------------------------------------------------
// Harmony-at-time and chroma rotation helpers
// ---------------------------------------------------------------------------

/// If the track modulates inside the overlap window, report the *later* key so
/// compatibility is judged against what actually plays during the mix.
fn key_at(h: &TrackHarmony, t: f64) -> TrackHarmony {
    let mut hh = h.clone();
    for m in &h.modulations {
        if m.time <= t {
            let (name, camelot, scale) = parse_key_tag(Some(&m.to_key));
            hh.key = name;
            hh.camelot = camelot;
            hh.scale = scale;
        }
    }
    hh
}

fn rotate_chroma(c: &[f64; 12], semitones: i32) -> [f64; 12] {
    let mut out = [0.0f64; 12];
    let s = ((semitones % 12) + 12) % 12;
    for i in 0..12 {
        out[(i + s as usize) % 12] = c[i];
    }
    out
}

// ---------------------------------------------------------------------------
// Automation generation (spec section 5 rules)
// ---------------------------------------------------------------------------

/// Produce automation lanes for a transition type honoring the mixing rules:
/// 1 never both basses full, 2 equal-power volume, 3 complementary bass swap,
/// 4 duck mid on vocal clash, 5 shorter overlap handled by planner already,
/// 6 filter/echo instead of long blend at low beat confidence, 7 limiter in mix.
pub fn generate_automation(
    ty: TransitionType,
    dur: f64,
    a: &TrackAnalysis,
    b: &TrackAnalysis,
    c: &Candidate,
    _pref: UserMixPreference,
) -> Vec<AutomationCurve> {
    use MixParam::*;
    use TrackRole::*;
    let mut lanes = Vec::new();
    let ap = |time: f64, value: f64, curve: CurveShape| AutomationPoint { time, value, curve };

    // Rule 2: equal-power volume crossfade always present (except quick cut).
    if ty != TransitionType::QuickCut {
        lanes.push(AutomationCurve::new(
            Volume,
            Source,
            vec![ap(0.0, 1.0, CurveShape::Linear), ap(dur, 0.0, CurveShape::EqualPower)],
        ));
        lanes.push(AutomationCurve::new(
            Volume,
            Target,
            vec![ap(0.0, 0.0, CurveShape::Linear), ap(dur, 1.0, CurveShape::EqualPower)],
        ));
    }

    match ty {
        TransitionType::SeamlessBeatmatchedBlend | TransitionType::BassSwap | TransitionType::DrumOnlyBridge => {
            // Rules 1 & 3: complementary bass curves — source low exits during
            // the first third, target low enters during the second third.
            lanes.push(AutomationCurve::new(
                Low,
                Source,
                vec![ap(0.0, 1.0, CurveShape::Linear), ap(dur * 0.33, 0.0, CurveShape::SCurve), ap(dur, 0.0, CurveShape::Linear)],
            ));
            lanes.push(AutomationCurve::new(
                Low,
                Target,
                vec![ap(0.0, 0.0, CurveShape::Linear), ap(dur * 0.33, 0.0, CurveShape::Linear), ap(dur * 0.66, 1.0, CurveShape::SCurve)],
            ));
            if ty == TransitionType::DrumOnlyBridge {
                // melodic layers of the target enter late
                lanes.push(AutomationCurve::new(
                    Mid,
                    Target,
                    vec![ap(0.0, 0.0, CurveShape::Linear), ap(dur * 0.5, 0.0, CurveShape::Linear), ap(dur, 1.0, CurveShape::SCurve)],
                ));
                lanes.push(AutomationCurve::new(
                    High,
                    Target,
                    vec![ap(0.0, 0.3, CurveShape::Linear), ap(dur, 1.0, CurveShape::EqualPower)],
                ));
            }
            // Rule 4: duck the mid of whichever side has lower vocal priority
            if c.cost.vocal_collision_penalty > 0.2 {
                let voc_a = mean_band(&a.spectral.vocal_presence, c.exit_t, dur);
                let voc_b = mean_band(&b.spectral.vocal_presence, c.entry_t, dur);
                let loser = if voc_a >= voc_b { Target } else { Source };
                let t0 = if loser == Source { dur * 0.6 } else { dur * 0.1 };
                let t1 = if loser == Source { dur } else { dur * 0.5 };
                let (start_v, end_v) = if loser == Source { (1.0, 0.5) } else { (0.5, 1.0) };
                lanes.push(AutomationCurve::new(
                    Mid,
                    loser,
                    vec![ap(0.0, start_v, CurveShape::Linear), ap(t0, start_v, CurveShape::Linear), ap(t1, end_v, CurveShape::SCurve), ap(dur, if loser == Source { 0.5 } else { 1.0 }, CurveShape::Linear)],
                ));
            }
        }
        TransitionType::FilterSweep => {
            // HP on source rising, LP on target opening up (energy build).
            lanes.push(AutomationCurve::new(
                Filter,
                Source,
                vec![ap(0.0, 0.02, CurveShape::Linear), ap(dur, 0.95, CurveShape::Expo)],
            ));
            lanes.push(AutomationCurve::new(
                Filter,
                Target,
                vec![ap(0.0, 0.95, CurveShape::Linear), ap(dur, 0.02, CurveShape::SCurve)],
            ));
            lanes.push(AutomationCurve::new(
                Low,
                Source,
                vec![ap(0.0, 1.0, CurveShape::Linear), ap(dur * 0.5, 0.0, CurveShape::Expo)],
            ));
        }
        TransitionType::EchoOut => {
            // Rule 6: mask conflicts with delay/reverb tail on the source.
            lanes.push(AutomationCurve::new(
                Delay,
                Source,
                vec![ap(0.0, 0.0, CurveShape::Linear), ap(dur * 0.4, 0.8, CurveShape::Expo), ap(dur, 1.0, CurveShape::Linear)],
            ));
            lanes.push(AutomationCurve::new(
                Reverb,
                Source,
                vec![ap(0.0, 0.0, CurveShape::Linear), ap(dur, 0.6, CurveShape::Expo)],
            ));
            lanes.push(AutomationCurve::new(
                Low,
                Source,
                vec![ap(0.0, 1.0, CurveShape::Linear), ap(dur * 0.5, 0.0, CurveShape::Expo)],
            ));
            lanes.push(AutomationCurve::new(
                Low,
                Target,
                vec![ap(0.0, 0.0, CurveShape::Linear), ap(dur * 0.6, 1.0, CurveShape::SCurve)],
            ));
        }
        TransitionType::QuickCut | TransitionType::MetadataFallbackCrossfade => {
            // nothing extra; metadata fallback uses plain equal-power above
        }
    }
    lanes
}

fn mean_band(v: &[f64], start: f64, dur: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    let n = v.len().max(1);
    let i0 = ((start / 0.1) as usize).min(n - 1);
    let i1 = (i0 + (dur / 0.1) as usize).min(n - 1);
    if i1 <= i0 {
        return v[i0];
    }
    v[i0..i1].iter().sum::<f64>() / (i1 - i0) as f64
}

// ---------------------------------------------------------------------------
// Online weight auto-calibration (spec §3 last paragraph)
// ---------------------------------------------------------------------------

/// Outcome signal fed back after each rendered transition.
#[derive(Clone, Copy, Debug, Default)]
pub struct TransitionOutcome {
    /// Host observed user skipping during the transition (implicit negative).
    pub skipped: bool,
    /// Measured RMS jump in LU post-render.
    pub rms_jump_lu: f64,
    /// Measured spectral-flux discontinuity at the seam (0..1).
    pub flux_discontinuity: f64,
    /// Clipped samples produced despite the limiter (0 = none).
    pub clip_events: usize,
}

/// Multiplicative exponentiated-gradient recalibration of the weight vector:
/// terms whose observable proxies got worse are nudged up, others decay back
/// toward the prior, then the vector is renormalized onto Σw = const simplex.
pub fn calibrate_weights(weights: &mut CostWeights, o: &TransitionOutcome, eta: f64) {
    let prior = CostWeights::default();
    let upd = |w: &mut f64, base: f64, obs: f64| {
        // obs ∈ [0,1]: 0 = this term behaved fine, 1 = it caused the failure
        let g = (obs - 0.25).max(0.0); // deadband: don't chase noise
        *w *= (eta * g).exp();
        // pull back toward prior slowly so calibration doesn't drift forever
        *w = 0.98 * *w + 0.02 * base;
        *w = w.clamp(base * 0.25, base * 4.0);
    };
    let bad = if o.skipped { 1.0 } else { 0.0 };
    let rough = (o.rms_jump_lu / 6.0).min(1.0);
    let dirty = (o.flux_discontinuity).min(1.0);
    let clip = if o.clip_events > 0 { 1.0 } else { 0.0 };

    upd(&mut weights.w1_beat_alignment, prior.w1_beat_alignment, bad);
    upd(&mut weights.w2_key_conflict, prior.w2_key_conflict, bad * 0.8);
    upd(&mut weights.w3_phrase_misalignment, prior.w3_phrase_misalignment, bad * 0.6);
    upd(&mut weights.w4_bass_conflict, prior.w4_bass_conflict, dirty.max(clip));
    upd(&mut weights.w5_vocal_collision, prior.w5_vocal_collision, bad * 0.7);
    upd(&mut weights.w6_energy_jump, prior.w6_energy_jump, rough);
    upd(&mut weights.w7_structural_mismatch, prior.w7_structural_mismatch, bad * 0.5);
    upd(&mut weights.w8_artifact, prior.w8_artifact, clip.max(dirty * 0.5));
    // bonuses only get stronger when transitions were *not* skipped
    let good = 1.0 - bad;
    weights.w9_smoothness_bonus = (weights.w9_smoothness_bonus * (1.0 + 0.02 * good)).min(prior.w9_smoothness_bonus * 2.0);
    weights.w10_musical_interest_bonus =
        (weights.w10_musical_interest_bonus * (1.0 + 0.02 * good)).min(prior.w10_musical_interest_bonus * 2.0);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth;

    fn analyze(o: &SynthOptions) -> TrackAnalysis {
        let buf = synth::synth_track(o);
        let mut eng = crate::analysis::AnalysisEngine::default();
        eng.analyze(&buf)
    }

    #[test]
    fn same_key_same_bpm_scores_better_than_distant_key() {
        let a = analyze(&SynthOptions { bpm: 124.0, key_pc: name_to_pc_i("A"), minor: true, duration_s: 40.0, ..Default::default() });
        let b_near = analyze(&SynthOptions { bpm: 124.0, key_pc: name_to_pc_i("A"), minor: true, duration_s: 40.0, seed: 7, ..Default::default() });
        let b_far = analyze(&SynthOptions { bpm: 124.0, key_pc: name_to_pc_i("F#"), minor: true, duration_s: 40.0, seed: 7, ..Default::default() });
        let p = TransitionPlanner::default();
        let near = p.compatibility(&a, &b_near).overall;
        let far = p.compatibility(&a, &b_far).overall;
        assert!(near > far, "near {near} should beat far {far}");
    }

    #[test]
    fn big_bpm_gap_yields_quick_cut_or_fallback() {
        let a = analyze(&SynthOptions { bpm: 128.0, duration_s: 40.0, ..Default::default() });
        let b = analyze(&SynthOptions { bpm: 150.0, duration_s: 40.0, seed: 3, ..Default::default() });
        let p = TransitionPlanner::default();
        let plan = p.plan_with_fallback(&a, &b, &PlaybackContext::default(), UserMixPreference::Smooth);
        assert!(
            matches!(
                plan.transition_type,
                TransitionType::QuickCut | TransitionType::EchoOut | TransitionType::FilterSweep | TransitionType::MetadataFallbackCrossfade
            ),
            "unexpected {:?}",
            plan.transition_type
        );
    }

    #[test]
    fn seamless_blend_has_complementary_bass_lanes() {
        let a = analyze(&SynthOptions { bpm: 124.0, duration_s: 40.0, ..Default::default() });
        let b = analyze(&SynthOptions { bpm: 124.0, duration_s: 40.0, seed: 11, ..Default::default() });
        let p = TransitionPlanner::default();
        let plan = p.plan_with_fallback(&a, &b, &PlaybackContext::default(), UserMixPreference::Smooth);
        // whichever type won, low lanes must never be simultaneously ≥0.9
        let lows: Vec<&AutomationCurve> = plan.automation.iter().filter(|l| l.parameter == MixParam::Low).collect();
        if lows.len() == 2 {
            let steps = 50;
            for s in 0..=steps {
                let t = plan.duration * s as f64 / steps as f64;
                let x = lows[0].eval(t) + lows[1].eval(t);
                assert!(x <= 1.95, "both basses open at t={t}: sum {x}");
            }
        }
        assert!(plan.confidence > 0.0 && plan.confidence <= 1.0);
        assert!(plan.fallback_plan.is_some());
    }

    #[test]
    fn metadata_only_pair_falls_back_to_crossfade() {
        let md = TrackMetadata { id: "m".into(), declared_bpm: Some(122.0), declared_key: Some("8A".into()), ..Default::default() };
        let a = TrackAnalysis::from_metadata(md.clone(), 180.0);
        let b = TrackAnalysis::from_metadata(md, 200.0);
        let p = TransitionPlanner::default();
        let plan = p.plan_with_fallback(&a, &b, &PlaybackContext::default(), UserMixPreference::Invisible);
        assert_eq!(plan.transition_type, TransitionType::MetadataFallbackCrossfade);
        assert!(!plan.beatmatch);
    }

    #[test]
    fn calibration_moves_weight_after_skip() {
        let mut w = CostWeights::default();
        let before = w.w1_beat_alignment;
        calibrate_weights(&mut w, &TransitionOutcome { skipped: true, ..Default::default() }, 0.3);
        assert!(w.w1_beat_alignment > before);
    }

    fn name_to_pc_i(name: &str) -> usize {
        crate::synth::name_to_pc(name)
    }
}
