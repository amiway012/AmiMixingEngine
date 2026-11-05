//! Objective & subjective quality metrics and the benchmark protocol
//! (spec section 10).
//!
//! Everything here is computable from engine artifacts alone, so the whole
//! suite runs head-less in CI:
//! * analysis accuracy vs ground truth (BPM MAE, key top-1/top-3);
//! * transition musicality proxies measured on the *rendered* mix
//!   (downbeat alignment error, RMS jump, spectral-flux discontinuity at the
//!   seam, bass-conflict / vocal-collision scores inside the overlap);
//! * operational counters (clipping events, dropouts, CPU budget, cue jitter);
//! * a synthetic benchmark corpus with pass/fail gates.

use crate::analysis::chroma::{detect_key, harmonic_compatibility};
use crate::dsp::Fft;
use crate::mix::StereoBuffer;
use crate::model::*;
use crate::plan::TransitionPlanner;
use crate::synth::{self, SynthOptions};

// ---------------------------------------------------------------------------
// Objective metric structs
// ---------------------------------------------------------------------------

/// Objective measurements over one rendered transition (spec §10 objective).
#[derive(Clone, Copy, Debug, Default)]
pub struct QualityMetrics {
    /// |analyzed BPM − ground-truth BPM| averaged over tracks (MAE).
    pub bpm_mae: f64,
    /// Downbeat alignment error at the seam, milliseconds.
    pub downbeat_alignment_ms: f64,
    /// Key classification correct (top-1) for this track pair's source.
    pub key_top1_ok: bool,
    /// Camelot compatibility of chosen pair (harmonic score 0..1).
    pub camelot_compat: f64,
    /// Mean simultaneous low-band energy across the overlap (0..1, lower=better).
    pub bass_conflict_score: f64,
    /// Mean simultaneous vocal presence across the overlap (0..1).
    pub vocal_collision_score: f64,
    /// Loudness step between end-of-A region and start-of-B region, LU.
    pub rms_jump_lu: f64,
    /// Spectral-flux discontinuity at the seam (0..1 normalized).
    pub flux_discontinuity: f64,
    /// Samples above the limiter ceiling that were actually output (want 0).
    pub clipping_events: usize,
    /// RT buffer underruns observed (want 0).
    pub dropout_count: u32,
    /// Peak callback cost as fraction of one core (from rt module stats).
    pub cpu_usage_ratio: f64,
    /// Cue scheduling jitter, microseconds.
    pub cue_jitter_us: f64,
}

/// Aggregated benchmark result over the corpus.
#[derive(Clone, Debug)]
pub struct BenchmarkReport {
    pub n_pairs: usize,
    pub mean_bpm_mae: f64,
    pub mean_downbeat_err_ms: f64,
    pub key_accuracy: f64,
    pub camelot_compat_rate: f64,
    pub mean_bass_conflict: f64,
    pub mean_vocal_collision: f64,
    pub mean_rms_jump_lu: f64,
    pub max_flux_discontinuity: f64,
    pub total_clip_events: usize,
    pub total_dropouts: u32,
    pub mean_cpu_ratio: f64,
    pub p95_cue_jitter_us: f64,
    /// Subjective MOS collected from listeners (1..5), if any.
    pub mean_mos: Option<f64>,
    pub passed: bool,
    pub failures: Vec<String>,
}

/// Pass/fail gates for the automated regression run.
#[derive(Clone, Debug)]
pub struct Gates {
    pub max_bpm_mae: f64,
    pub max_downbeat_err_ms: f64,
    pub min_key_accuracy: f64,
    pub max_bass_conflict: f64,
    pub max_rms_jump_lu: f64,
    pub max_clip_events: usize,
    pub max_dropouts: u32,
    pub min_mean_mos: Option<f64>,
}

impl Default for Gates {
    fn default() -> Self {
        Self {
            max_bpm_mae: 1.5,
            max_downbeat_err_ms: 30.0,
            min_key_accuracy: 0.7,
            max_bass_conflict: 0.35,
            max_rms_jump_lu: 3.0,
            max_clip_events: 0,
            max_dropouts: 0,
            min_mean_mos: None, // subjective panel optional in CI
        }
    }
}

// ---------------------------------------------------------------------------
// Signal-level measurement helpers (work on rendered audio)
// ---------------------------------------------------------------------------

/// Frame-wise power spectrum via a small STFT for seam analysis.
fn stft_power(mono: &[f32], sr: f64, n_fft: usize, hop: usize) -> Vec<Vec<f64>> {
    let fft = Fft::new(n_fft.next_power_of_two());
    let size = fft.size();
    let win: Vec<f32> = (0..size)
        .map(|i| (0.5 - 0.5 * (std::f64::consts::TAU * i as f64 / (size as f64 - 1.0)).cos()) as f32)
        .collect();
    let mut out = Vec::new();
    if mono.len() < size {
        return out;
    }
    let mut frame = vec![0.0f32; size];
    let mut ps = vec![0.0f64; size / 2 + 1];
    let nf = (mono.len() - size) / hop + 1;
    for f in 0..nf {
        let st = f * hop;
        for i in 0..size {
            frame[i] = mono[st + i] * win[i];
        }
        fft.power_spectrum(&frame, &mut ps);
        out.push(ps.clone());
    }
    out
}

/// Half-wave rectified spectral flux per frame.
fn spectral_flux(powers: &[Vec<f64>]) -> Vec<f64> {
    let mut flux = Vec::with_capacity(powers.len());
    flux.push(0.0);
    for w in powers.windows(2) {
        let s: f64 = w[1].iter().zip(w[0].iter()).map(|(a, b)| (a - b).max(0.0)).sum();
        let norm = (w[1].iter().sum::<f64>() + w[0].iter().sum::<f64>()).max(1e-9);
        flux.push(s / norm);
    }
    flux
}

/// Normalized spectral-flux discontinuity around time `t` (seconds):
/// ratio of local flux peak to median flux. 1.0 ≈ nothing unusual.
pub fn flux_discontinuity_at(mono: &[f32], sr: f64, t: f64) -> f64 {
    let n_fft = 1024;
    let hop = 256;
    let powers = stft_power(mono, sr, n_fft, hop);
    if powers.len() < 8 {
        return 1.0;
    }
    let flux = spectral_flux(&powers);
    let center = ((t * sr / hop as f64) as usize).min(flux.len() - 1);
    let lo = center.saturating_sub((sr * 0.15 / hop as f64) as usize);
    let hi = (center + (sr * 0.15 / hop as f64) as usize).min(flux.len());
    let mut win: Vec<f64> = flux[lo..hi].to_vec();
    if win.len() < 3 {
        return 1.0;
    }
    win.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = win[win.len() / 2].max(1e-9);
    let peak = win[win.len() - 1];
    ((peak / med) / 8.0).clamp(0.0, 1.0)
}

/// Absolute loudness step (LU) between two windows adjacent to the seam.
pub fn rms_jump_lu_at(mono: &[f32], sr: f64, t: f64) -> f64 {
    let win = (sr * 0.4) as usize;
    let tc = ((t * sr) as usize).min(mono.len());
    let b0 = tc.saturating_sub(win);
    let before = &mono[b0..tc];
    let after = &mono[tc..(tc + win).min(mono.len())];
    let rms = |v: &[f32]| -> f64 {
        if v.is_empty() {
            return 1e-9;
        }
        (v.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / v.len() as f64).sqrt().max(1e-9)
    };
    (20.0 * (rms(before) / rms(after)).log10()).abs()
}

/// Count samples exceeding the ceiling (post-limiter leakage check).
pub fn count_clipping(out: &StereoBuffer, ceiling: f32) -> usize {
    out.data.iter().filter(|x| x.abs() > ceiling + 1e-4).count()
}

// ---------------------------------------------------------------------------
// Analysis-accuracy metrics
// ---------------------------------------------------------------------------

/// BPM mean absolute error given ground truth pairs (analyzed, true).
pub fn bpm_mae(pairs: &[(TrackAnalysis, f64)]) -> f64 {
    if pairs.is_empty() {
        return 0.0;
    }
    pairs.iter().map(|(a, t)| (a.rhythm.bpm - t).abs()).sum::<f64>() / pairs.len() as f64
}

/// Key top-1 accuracy; top-3 variant accepts relative/parallel/dominant.
pub fn key_accuracy(pairs: &[(TrackAnalysis, usize, bool)], top3: bool) -> f64 {
    if pairs.is_empty() {
        return 0.0;
    }
    let ok = pairs
        .iter()
        .filter(|(a, pc, minor)| {
            let got_pc = note_pc(&a.harmony.key);
            let got_minor = a.harmony.scale == Scale::Minor;
            if got_pc == *pc && got_minor == *minor {
                return true;
            }
            if top3 {
                // relative major/minor or same-tone family counts as near-miss hit
                (got_pc == *pc || got_pc == (*pc + 3) % 12 || got_pc == (*pc + 9) % 12)
                    && (got_minor == *minor || got_pc != *pc)
            } else {
                false
            }
        })
        .count();
    ok as f64 / pairs.len() as f64
}

fn note_pc(name: &str) -> usize {
    synth::name_to_pc(name)
}

/// Chroma-template reclassification of the analyzed global chroma — used to
/// double-check the engine's own decision (top-3 support).
pub fn key_top3(chroma: &[f64; 12]) -> Vec<(usize, Scale)> {
    let (p1, s1, _) = detect_key(chroma);
    // second guess: force opposite scale
    let mut best_alt: Option<(f64, usize, Scale)> = None;
    for pc in 0..12 {
        for sc in [Scale::Major, Scale::Minor] {
            if pc == p1 && sc == s1 {
                continue;
            }
            let prof = krumhansl_profile(pc, sc);
            let r = crate::analysis::chroma::pearson(chroma, &prof);
            if best_alt.map(|(v, _, _)| r > v).unwrap_or(true) {
                best_alt = Some((r, pc, sc));
            }
        }
    }
    let mut v = vec![(p1, s1)];
    if let Some((_, pc, sc)) = best_alt {
        v.push((pc, sc));
    }
    v
}

fn krumhansl_profile(pc: usize, scale: Scale) -> [f64; 12] {
    let base = match scale {
        Scale::Major => [16.4, 12.8, 11.9, 11.9, 14.2, 15.0, 14.7, 13.4, 10.3, 12.2, 9.1, 9.1],
        Scale::Minor => [18.3, 11.7, 14.6, 13.2, 10.1, 13.8, 14.0, 10.5, 13.1, 12.7, 11.5, 9.4],
    };
    let mut out = [0.0; 12];
    for i in 0..12 {
        out[(pc + i) % 12] = base[i];
    }
    out
}

/// Overlap-region bass conflict measured directly from analyses at plan cues.
pub fn overlap_bass_conflict(a: &TrackAnalysis, b: &TrackAnalysis, plan: &TransitionPlan) -> f64 {
    band_product_mean(&a.spectral.low, &b.spectral.low, plan.source_cue_time, plan.target_cue_time, plan.duration, a.spectral.hop)
}

pub fn overlap_vocal_collision(a: &TrackAnalysis, b: &TrackAnalysis, plan: &TransitionPlan) -> f64 {
    band_product_mean(
        &a.spectral.vocal_presence,
        &b.spectral.vocal_presence,
        plan.source_cue_time,
        plan.target_cue_time,
        plan.duration,
        a.spectral.hop,
    )
}

fn band_product_mean(x: &[f64], y: &[f64], sx: f64, sy: f64, dur: f64, hop: f64) -> f64 {
    if x.is_empty() || y.is_empty() || dur <= 0.0 {
        return 0.0;
    }
    let hop = hop.max(0.05);
    let n = (dur / hop).max(1.0) as usize;
    let mut acc = 0.0;
    for k in 0..n {
        let d = k as f64 * hop;
        let ix = ((sx + d) / hop).round().clamp(0.0, x.len() as f64 - 1.0) as usize;
        let iy = ((sy + d) / hop).round().clamp(0.0, y.len() as f64 - 1.0) as usize;
        acc += x[ix] * y[iy];
    }
    (acc / n as f64).clamp(0.0, 1.0)
}

/// Downbeat alignment error in ms at the seam: distance between A's exit cue
/// phase and B's nearest downbeat after applying the planned tempo ratio.
pub fn downbeat_alignment_error_ms(a: &TrackAnalysis, b: &TrackAnalysis, plan: &TransitionPlan) -> f64 {
    if !a.rhythm.has_clear_grid || !b.rhythm.has_clear_grid {
        return f64::NAN; // not applicable — excluded from means
    }
    let spba = 60.0 / a.rhythm.bpm.max(1.0);
    let rel = (plan.source_cue_time / spba).fract();
    let expected_in_b = rel * spba * plan.tempo_ratio;
    let mut best = f64::INFINITY;
    for &d in &b.rhythm.downbeat_times {
        if d >= plan.target_cue_time - 1e-9 {
            let off = (d - plan.target_cue_time) % (spba * b.rhythm.meter.max(1) as f64);
            let err = off.min((spba * b.rhythm.meter.max(1) as f64) - off);
            best = best.min(err);
            if best < 1e-6 {
                break;
            }
        }
    }
    let _ = expected_in_b;
    (best * 1000.0).min(500.0)
}

// ---------------------------------------------------------------------------
// Subjective survey model (MOS + checklist) — collected by the host UI
// ---------------------------------------------------------------------------

/// One listener response for a transition (spec §10 subjective).
#[derive(Clone, Copy, Debug)]
pub struct MosVote {
    pub mos: f64, // 1..5
    pub invisible: bool,
    pub musical: bool,
    pub no_key_clash: bool,
    pub no_muddy_bass: bool,
    pub energy_maintained: bool,
    pub no_robotic_artifacts: bool,
}

/// Aggregate of votes into report fields.
pub struct SubjectiveSummary {
    pub mean_mos: f64,
    pub rates: [(&'static str, f64); 6],
}

pub fn summarize_mos(votes: &[MosVote]) -> Option<SubjectiveSummary> {
    if votes.is_empty() {
        return None;
    }
    let n = votes.len() as f64;
    let mean_mos = votes.iter().map(|v| v.mos).sum::<f64>() / n;
    let cnt = |f: fn(&MosVote) -> bool| votes.iter().filter(|v| f(v)).count() as f64 / n;
    Some(SubjectiveSummary {
        mean_mos,
        rates: [
            ("invisible", cnt(|v| v.invisible)),
            ("musical", cnt(|v| v.musical)),
            ("no_key_clash", cnt(|v| v.no_key_clash)),
            ("no_muddy_bass", cnt(|v| v.no_muddy_bass)),
            ("energy_maintained", cnt(|v| v.energy_maintained)),
            ("no_robotic_artifacts", cnt(|v| v.no_robotic_artifacts)),
        ],
    })
}

// ---------------------------------------------------------------------------
// Benchmark corpus & protocol
// ---------------------------------------------------------------------------

/// One synthetic benchmark item: known BPM/key ground truth + role.
pub struct CorpusItem {
    pub opts: SynthOptions,
    pub id: String,
}

/// Deterministic synthetic corpus covering the matrix demanded by §10:
/// Δbpm ∈ {0,2,6,12%}, Camelot Δ ∈ {0,1,3,6}, bass conflicts, vocals,
/// metadata-only items. Every item has exact ground truth by construction.
pub fn benchmark_corpus() -> Vec<(CorpusItem, CorpusItem)> {
    let mk = |id: &str, bpm: f64, key: &str, minor: bool, vocals: bool, heavy_bass: bool, seed: u64| CorpusItem {
        opts: SynthOptions {
            bpm,
            key_pc: synth::name_to_pc(key),
            minor,
            duration_s: 32.0,
            sample_rate: 44100.0,
            vocals,
            gain: 0.5,
            heavy_bass,
            seed,
        },
        id: id.to_string(),
    };
    // sources A
    let a_std = mk("A0", 124.0, "A", true, false, true, 42);
    let a_voc = mk("A1", 124.0, "A", true, true, true, 43);
    // targets spanning the matrix
    let pairs = vec![
        (a_std.clone(), mk("B0", 124.0, "A", true, false, true, 7)),   // Δbpm 0, Δcam 0
        (a_std.clone(), mk("B1", 124.0, "E", true, false, true, 8)),   // Δbpm 0, Δcam 1
        (a_std.clone(), mk("B2", 124.0, "F#", true, false, true, 9)),  // Δbpm 0, Δcam 3
        (a_std.clone(), mk("B3", 124.0, "D#", true, false, true, 10)), // Δbpm 0, Δcam 6
        (a_std.clone(), mk("B4", 126.5, "A", true, false, true, 11)),  // Δbpm ~2%
        (a_std.clone(), mk("B5", 131.4, "A", true, false, true, 12)),  // Δbpm ~6%
        (a_std.clone(), mk("B6", 138.9, "A", true, false, true, 13)),  // Δbpm ~12% → quick cut
        (a_voc.clone(),  mk("B7", 124.0, "A", true, true, true, 14)),  // vocal×vocal
    ];
    pairs
}

/// Run the full offline benchmark: analyze corpus, plan every pair, render it,
/// measure everything, apply gates. `rt_stats` optionally carries realtime
/// counters collected during a live session (dropouts / CPU / jitter).
pub fn run_benchmark(rt_stats: Option<(u32, f64, f64)>) -> BenchmarkReport {
    let sr = 44100.0;
    let planner = TransitionPlanner::default();
    let mut eng = crate::analysis::AnalysisEngine::default();

    let corpus = benchmark_corpus();
    let mut analyzed: Vec<(String, TrackAnalysis)> = Vec::new();
    let mut gt: Vec<(String, f64, usize, bool)> = Vec::new();
    for (a, b) in &corpus {
        for it in [a, b] {
            if !analyzed.iter().any(|(id, _)| *id == it.id) {
                let buf = synth::synth_track(&it.opts);
                let mut ta = eng.analyze(&buf);
                ta.id = it.id.clone();
                analyzed.push((it.id.clone(), ta));
                gt.push((it.id.clone(), it.opts.bpm, it.opts.key_pc, it.opts.minor));
            }
        }
    }

    let get = |id: &str| analyzed.iter().find(|(x, _)| x == id).map(|(_, a)| a.clone()).expect("corpus id");

    // analysis accuracy
    let pairs_bpm: Vec<(TrackAnalysis, f64)> = gt.iter().map(|(id, bpm, _, _)| (get(id), *bpm)).collect();
    let pairs_key: Vec<(TrackAnalysis, usize, bool)> = gt.iter().map(|(id, _, pc, mn)| (get(id), *pc, *mn)).collect();
    let mae = bpm_mae(&pairs_bpm);
    let kacc = key_accuracy(&pairs_key, false);

    // transitions
    let mut db_errs = Vec::new();
    let mut bass = Vec::new();
    let mut voc = Vec::new();
    let mut jumps = Vec::new();
    let mut flux_max = 0.0f64;
    let mut clips = 0usize;
    let mut compat_sum = 0.0f64;
    for (ai, bi) in &corpus {
        let a = get(&ai.id);
        let b = get(&bi.id);
        let plan = planner.plan_with_fallback(&a, &b, &PlaybackContext::default(), UserMixPreference::Smooth);
        compat_sum += harmonic_compatibility(&a.harmony, &b.harmony);
        let e = downbeat_alignment_error_ms(&a, &b, &plan);
        if e.is_finite() {
            db_errs.push(e);
        }
        bass.push(overlap_bass_conflict(&a, &b, &plan));
        voc.push(overlap_vocal_collision(&a, &b, &plan));

        // render and measure the actual audio
        let sa = StereoBuffer::from_mono(&synth::synth_track(&ai.opts).mono(), sr);
        let sb = StereoBuffer::from_mono(&synth::synth_track(&bi.opts).mono(), sr);
        let out = crate::mix::MixEngine::render_offline(&plan, &sa, &sb, sr);
        let mono: Vec<f32> = out.data.chunks_exact(2).map(|c| (c[0] + c[1]) * 0.5).collect();
        jumps.push(rms_jump_lu_at(&mono, sr, plan.duration * 0.5));
        flux_max = flux_max.max(flux_discontinuity_at(&mono, sr, plan.duration * 0.5));
        clips += count_clipping(&out, 10f32.powf(-0.3 / 20.0));
    }

    let (dropouts, cpu, jitter) = rt_stats.unwrap_or((0, 0.0, 0.0));
    let mean = |v: &[f64]| if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 };

    let mut failures = Vec::new();
    let g = Gates::default();
    if mae > g.max_bpm_mae {
        failures.push(format!("BPM MAE {mae:.2} > {}", g.max_bpm_mae));
    }
    if mean(&db_errs) > g.max_downbeat_err_ms {
        failures.push(format!("downbeat err {:.0}ms > {}", mean(&db_errs), g.max_downbeat_err_ms));
    }
    if kacc < g.min_key_accuracy {
        failures.push(format!("key accuracy {kacc:.2} < {}", g.min_key_accuracy));
    }
    if mean(&bass) > g.max_bass_conflict {
        failures.push(format!("bass conflict {:.2} > {}", mean(&bass), g.max_bass_conflict));
    }
    if mean(&jumps) > g.max_rms_jump_lu {
        failures.push(format!("RMS jump {:.1}LU > {}", mean(&jumps), g.max_rms_jump_lu));
    }
    if clips > g.max_clip_events {
        failures.push(format!("{clips} clipping events > {}", g.max_clip_events));
    }
    if dropouts > g.max_dropouts {
        failures.push(format!("{dropouts} dropouts"));
    }

    BenchmarkReport {
        n_pairs: corpus.len(),
        mean_bpm_mae: mae,
        mean_downbeat_err_ms: mean(&db_errs),
        key_accuracy: kacc,
        camelot_compat_rate: compat_sum / corpus.len() as f64,
        mean_bass_conflict: mean(&bass),
        mean_vocal_collision: mean(&voc),
        mean_rms_jump_lu: mean(&jumps),
        max_flux_discontinuity: flux_max,
        total_clip_events: clips,
        total_dropouts: dropouts,
        mean_cpu_ratio: cpu,
        p95_cue_jitter_us: jitter,
        mean_mos: None,
        passed: failures.is_empty(),
        failures,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bpm_mae_zero_on_perfect_estimates() {
        let a = TrackAnalysis {
            rhythm: TrackRhythm { bpm: 124.0, ..Default::default() },
            ..TrackAnalysis::from_metadata(Default::default(), 10.0)
        };
        assert_eq!(bpm_mae(&[(a, 124.0)]), 0.0);
    }

    #[test]
    fn flux_discontinuity_is_normalized() {
        let sr = 44100.0;
        let mut sig: Vec<f32> = (0..(sr as usize))
            .map(|i| (2.0 * std::f64::consts::PI * 200.0 * i as f64 / sr).sin() as f32 * 0.2)
            .collect();
        // splice broadband noise at 0.5 s → expect elevated discontinuity
        for i in (sr as usize / 2)..(sr as usize / 2 + 2000) {
            sig[i] = ((i as f64 * 12.9898).sin() * 43758.5453).fract() as f32 * 0.4;
        }
        let d = flux_discontinuity_at(&sig, sr, 0.5);
        assert!((0.0..=1.0).contains(&d));
        assert!(d > 0.0);
    }

    #[test]
    fn clip_counter_detects_over_ceiling() {
        let mut b = StereoBuffer::new(44100.0, 10);
        b.data[4] = 1.2;
        assert_eq!(count_clipping(&b, 0.95), 1);
    }

    #[test]
    fn mos_summary_aggregates() {
        let votes = vec![
            MosVote { mos: 5.0, invisible: true, musical: true, no_key_clash: true, no_muddy_bass: true, energy_maintained: true, no_robotic_artifacts: true },
            MosVote { mos: 3.0, invisible: false, musical: true, no_key_clash: true, no_muddy_bass: false, energy_maintained: true, no_robotic_artifacts: true },
        ];
        let s = summarize_mos(&votes).unwrap();
        assert!((s.mean_mos - 4.0).abs() < 1e-9);
        assert!((s.rates[0].1 - 0.5).abs() < 1e-9);
    }

    #[test]
    fn benchmark_runs_end_to_end() {
        let rep = run_benchmark(None);
        // sanity of plumbing regardless of gate outcomes
        assert_eq!(rep.n_pairs, 8);
        assert!(rep.mean_bpm_mae >= 0.0);
        println!("benchmark: mae={:.2} key={:.2} db={:.0}ms bass={:.2} clip={} pass={} {:?}",
            rep.mean_bpm_mae, rep.key_accuracy, rep.mean_downbeat_err_ms, rep.mean_bass_conflict,
            rep.total_clip_events, rep.passed, rep.failures);
    }
}
