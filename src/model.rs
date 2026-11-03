//! Data models for the whole engine (spec section 8), plus audio buffers.
//!
//! All times are in seconds, all energies on normalized linear scales unless
//! stated otherwise. Loudness values are LUFS (EBU R128).

use std::fmt;

// ---------------------------------------------------------------------------
// Audio representation
// ---------------------------------------------------------------------------

/// A decoded, sample-rate-normalized audio buffer. Interleaved stereo or mono.
#[derive(Clone, Debug)]
pub struct AudioBuffer {
    pub sample_rate: f64,
    /// Number of channels (1 = mono, 2 = interleaved stereo).
    pub channels: usize,
    /// Interleaved PCM samples in [-1.0, 1.0].
    pub samples: Vec<f32>,
}

impl AudioBuffer {
    pub fn new(sample_rate: f64, channels: usize, samples: Vec<f32>) -> Self {
        Self { sample_rate, channels, samples }
    }

    pub fn duration(&self) -> f64 {
        self.samples.len() as f64 / (self.sample_rate * self.channels as f64)
    }

    /// Mono downmix used for analysis.
    pub fn mono(&self) -> Vec<f32> {
        if self.channels == 1 {
            return self.samples.clone();
        }
        self.samples
            .chunks_exact(self.channels)
            .map(|c| c.iter().sum::<f32>() / c.len() as f32)
            .collect()
    }

    pub fn frame_at(&self, t: f64) -> usize {
        (t * self.sample_rate).clamp(0.0, self.n_frames() as f64) as usize
    }

    pub fn n_frames(&self) -> usize {
        self.samples.len() / self.channels
    }

    pub fn time_at(&self, frame: usize) -> f64 {
        frame as f64 / self.sample_rate
    }
}

/// Metadata accompanying a track (ID3 / Vorbis / iTunes tags …).
#[derive(Clone, Debug, Default)]
pub struct TrackMetadata {
    pub id: String,
    pub artist: String,
    pub title: String,
    pub album: String,
    pub genre: Option<String>,
    /// Declared BPM from metadata (may be absent or wrong).
    pub declared_bpm: Option<f64>,
    /// Declared key, e.g. "Am", "8A" Camelot, "F#m".
    pub declared_key: Option<String>,
    /// Energy 0..1 from external tagging services (optional).
    pub declared_energy: Option<f64>,
    pub extra_tags: Vec<(String, String)>,
}

// ---------------------------------------------------------------------------
// Stage 2 — rhythm
// ---------------------------------------------------------------------------

/// Output of the rhythmic analysis (spec stage 2).
#[derive(Clone, Debug, Default)]
pub struct TrackRhythm {
    pub bpm: f64,
    pub bpm_confidence: f64,
    pub beat_times: Vec<f64>,
    pub downbeat_times: Vec<f64>,
    /// Beats per bar (4 for 4/4, 3 for 3/4, 6 for 6/8 …).
    pub meter: usize,
    /// Tempo stability 0..1 (1 = metronomic, low = rubato/free).
    pub tempo_stability: f64,
    pub has_clear_grid: bool,
    /// Kick/transient strength per beat (0..1).
    pub beat_strength: Vec<f64>,
    /// Local tempo estimates for tracks with tempo changes (multi-BPM): (time, bpm).
    pub local_tempi: Vec<(f64, f64)>,
    /// Clear kick/snare pattern present (drum-machine-like material).
    pub has_clear_percussion: bool,
}

// ---------------------------------------------------------------------------
// Stage 3 — harmony
// ---------------------------------------------------------------------------

/// Musical mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scale {
    Major,
    Minor,
}

impl fmt::Display for Scale {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Scale::Major => write!(f, "major"),
            Scale::Minor => write!(f, "minor"),
        }
    }
}

/// Pitch class names, C = 0.
pub const NOTE_NAMES: [&str; 12] = [
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
];

pub fn pc_to_name(pc: usize) -> &'static str {
    NOTE_NAMES[pc % 12]
}

/// Intra-track modulation point.
#[derive(Clone, Debug)]
pub struct Modulation {
    pub time: f64,
    pub from_key: String,
    pub to_key: String,
}

/// Output of the tonal analysis (spec stage 3).
#[derive(Clone, Debug)]
pub struct TrackHarmony {
    pub key: String,
    pub scale: Scale,
    /// Camelot notation, e.g. "8A".
    pub camelot: String,
    pub confidence: f64,
    /// Averaged chroma profile, 12 bins, max-normalized.
    pub chroma: [f64; 12],
    pub modulations: Vec<Modulation>,
    /// Bass root pitch classes over time.
    pub bass_notes: Vec<f64>,
    /// Melodic density per window (0..1).
    pub melodic_density: Vec<f64>,
}

impl Default for TrackHarmony {
    fn default() -> Self {
        Self {
            key: "C".into(),
            scale: Scale::Major,
            camelot: "8B".into(),
            confidence: 0.0,
            chroma: [0.0; 12],
            modulations: Vec::new(),
            bass_notes: Vec::new(),
            melodic_density: Vec::new(),
        }
    }
}

/// Map (pitch class, scale) -> Camelot code (Mixed In Key convention:
/// Am = 8A, Cmaj = 8B).
pub fn camelot_for(pc: usize, scale: Scale) -> String {
    let minor_map: [usize; 12] = [5, 6, 7, 8, 9, 10, 11, 12, 1, 2, 3, 4];
    let major_map: [usize; 12] = [8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6, 7];
    match scale {
        Scale::Minor => format!("{}A", minor_map[pc % 12]),
        Scale::Major => format!("{}B", major_map[pc % 12]),
    }
}

/// Parse "8A"/"9B" into (number, letter).
pub fn parse_camelot(code: &str) -> Option<(usize, char)> {
    let code = code.trim();
    let (num, letter) = code.split_at(code.len().checked_sub(1)?);
    let n: usize = num.parse().ok()?;
    let l = letter.chars().next()?.to_ascii_uppercase();
    if (1..=12).contains(&n) && (l == 'A' || l == 'B') {
        Some((n, l))
    } else {
        None
    }
}

/// Circular distance between two Camelot codes on the wheel (0..6).
pub fn camelot_distance(a: &str, b: &str) -> usize {
    match (parse_camelot(a), parse_camelot(b)) {
        (Some((x, _)), Some((y, _))) => {
            let d = x.abs_diff(y);
            d.min(12 - d)
        }
        _ => 12,
    }
}

// ---------------------------------------------------------------------------
// Stage 4 — structure
// ---------------------------------------------------------------------------

/// Structural label for a segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SegmentLabel {
    Intro,
    Verse,
    Chorus,
    Drop,
    Breakdown,
    Bridge,
    Outro,
    Instrumental,
    Vocal,
    SilentGap,
    BuildUp,
    Tail,
    Unknown,
}

impl fmt::Display for SegmentLabel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            SegmentLabel::Intro => "intro",
            SegmentLabel::Verse => "verse",
            SegmentLabel::Chorus => "chorus",
            SegmentLabel::Drop => "drop",
            SegmentLabel::Breakdown => "breakdown",
            SegmentLabel::Bridge => "bridge",
            SegmentLabel::Outro => "outro",
            SegmentLabel::Instrumental => "instrumental",
            SegmentLabel::Vocal => "vocal",
            SegmentLabel::SilentGap => "silent_gap",
            SegmentLabel::BuildUp => "build_up",
            SegmentLabel::Tail => "tail",
            SegmentLabel::Unknown => "unknown",
        };
        write!(f, "{s}")
    }
}

/// One structural segment with descriptors.
#[derive(Clone, Debug)]
pub struct Segment {
    pub start: f64,
    pub end: f64,
    pub label: SegmentLabel,
    pub energy: f64,
    pub bass_energy: f64,
    pub mid_energy: f64,
    pub high_energy: f64,
    pub vocal_probability: f64,
    pub beat_strength: f64,
    pub key_confidence: f64,
}

/// Output of the structural analysis (spec stage 4).
#[derive(Clone, Debug, Default)]
pub struct TrackStructure {
    pub segments: Vec<Segment>,
    pub phrase_boundaries: Vec<f64>,
    pub safe_entry_points: Vec<f64>,
    pub safe_exit_points: Vec<f64>,
    /// True when real audio was analyzed (vs metadata-only guess).
    pub from_audio: bool,
}

// ---------------------------------------------------------------------------
// Stage 5 — spectral / energy profile
// ---------------------------------------------------------------------------

/// Band edges (Hz): low / low-mid / mid / high-mid / high.
pub const BAND_EDGES: [f64; 6] = [20.0, 120.0, 500.0, 2000.0, 6000.0, 16000.0];

/// Time-aligned spectral band energies (spec stage 5).
#[derive(Clone, Debug, Default)]
pub struct TrackSpectralProfile {
    pub time: Vec<f64>,
    /// 20–120 Hz.
    pub low: Vec<f64>,
    /// 120–500 Hz.
    pub low_mid: Vec<f64>,
    /// 500 Hz–2 kHz.
    pub mid: Vec<f64>,
    /// 2–6 kHz.
    pub high_mid: Vec<f64>,
    /// 6–16 kHz.
    pub high: Vec<f64>,
    pub rms: Vec<f64>,
    pub centroid: Vec<f64>,
    pub bass_conflict_risk: Vec<f64>,
    pub kick_density: Vec<f64>,
    pub vocal_presence: Vec<f64>,
    pub hop: f64,
}

impl TrackSpectralProfile {
    /// Index of the window covering `t`.
    pub fn idx_at(&self, t: f64) -> usize {
        if self.hop <= 0.0 || self.time.is_empty() {
            return 0;
        }
        ((t / self.hop).round() as isize).clamp(0, self.time.len() as isize - 1) as usize
    }
}

// ---------------------------------------------------------------------------
// Loudness & quality
// ---------------------------------------------------------------------------

/// EBU R128-style loudness measurements.
#[derive(Clone, Debug, Default)]
pub struct LoudnessProfile {
    pub integrated_lufs: f64,
    pub lra: f64,
    /// Short-term loudness curve (LUFS), ~0.4 s windows.
    pub short_term: Vec<f64>,
    pub true_peak: f64,
    /// Fraction of clipped samples 0..1.
    pub clip_ratio: f64,
    pub silent_regions: Vec<(f64, f64)>,
    pub fade_in_end: Option<f64>,
    pub fade_out_start: Option<f64>,
}

/// Trustworthiness of the analysis.
#[derive(Clone, Debug, Default)]
pub struct AnalysisQuality {
    pub score: f64,
    pub artifact_suspect: bool,
    pub source_clipping: bool,
    pub metadata_only: bool,
    pub duration_mismatch: bool,
    pub notes: Vec<String>,
}

/// Full per-track analysis bundle (spec section 8).
#[derive(Clone, Debug)]
pub struct TrackAnalysis {
    pub id: String,
    pub duration: f64,
    pub sample_rate: f64,
    pub metadata: TrackMetadata,
    pub rhythm: TrackRhythm,
    pub harmony: TrackHarmony,
    pub structure: TrackStructure,
    pub spectral: TrackSpectralProfile,
    pub loudness: LoudnessProfile,
    pub quality: AnalysisQuality,
}

impl TrackAnalysis {
    /// Metadata-only analysis when PCM is unavailable (spec stage 1 fallback).
    pub fn from_metadata(md: TrackMetadata, duration_hint: f64) -> Self {
        let bpm = md.declared_bpm.unwrap_or(120.0);
        let (key, camelot, scale) = parse_key_tag(md.declared_key.as_deref());
        let energy = md.declared_energy.unwrap_or(0.5);
        let n_windows = ((duration_hint / 0.5) as usize).max(1);
        let spectral = TrackSpectralProfile {
            time: (0..n_windows).map(|i| i as f64 * 0.5 + 0.25).collect(),
            low: vec![energy * 0.6; n_windows],
            low_mid: vec![energy * 0.5; n_windows],
            mid: vec![energy * 0.5; n_windows],
            high_mid: vec![energy * 0.4; n_windows],
            high: vec![energy * 0.3; n_windows],
            rms: vec![0.3; n_windows],
            centroid: vec![1500.0; n_windows],
            bass_conflict_risk: vec![energy * 0.5; n_windows],
            kick_density: vec![0.0; n_windows],
            vocal_presence: vec![0.0; n_windows],
            hop: 0.5,
        };
        let beat_period = 60.0 / bpm.max(1.0);
        let beats: Vec<f64> = (0..(duration_hint / beat_period) as usize)
            .map(|i| i as f64 * beat_period)
            .collect();
        let downbeats: Vec<f64> = beats.iter().step_by(4).cloned().collect();
        Self {
            id: md.id.clone(),
            duration: duration_hint,
            sample_rate: 0.0,
            metadata: md,
            rhythm: TrackRhythm {
                bpm,
                bpm_confidence: 0.35,
                beat_times: beats,
                downbeat_times: downbeats,
                meter: 4,
                tempo_stability: 0.5,
                has_clear_grid: false,
                ..Default::default()
            },
            harmony: TrackHarmony { key, scale, camelot, confidence: 0.3, ..Default::default() },
            structure: TrackStructure {
                segments: vec![Segment {
                    start: 0.0,
                    end: duration_hint,
                    label: SegmentLabel::Unknown,
                    energy,
                    bass_energy: energy * 0.6,
                    mid_energy: energy * 0.5,
                    high_energy: energy * 0.4,
                    vocal_probability: 0.0,
                    beat_strength: 0.5,
                    key_confidence: 0.3,
                }],
                phrase_boundaries: (0..)
                    .take_while(|i| (*i as f64) * 4.0 * beat_period < duration_hint)
                    .map(|i| (i as f64) * 4.0 * beat_period)
                    .collect(),
                safe_entry_points: vec![0.0],
                safe_exit_points: vec![duration_hint],
                from_audio: false,
            },
            spectral,
            loudness: LoudnessProfile {
                integrated_lufs: -14.0,
                true_peak: 0.9,
                ..Default::default()
            },
            quality: AnalysisQuality {
                score: 0.2,
                metadata_only: true,
                notes: vec!["metadata-only mode: equal-power crossfade fallback".into()],
                ..Default::default()
            },
        }
    }
}

/// Parse "Am", "A minor", "8A", "F#maj" into (name, camelot, scale).
pub fn parse_key_tag(tag: Option<&str>) -> (String, String, Scale) {
    let Some(t) = tag else {
        return ("C".into(), "8B".into(), Scale::Major);
    };
    let t = t.trim();
    if let Some((n, l)) = parse_camelot(t) {
        let scale = if l == 'A' { Scale::Minor } else { Scale::Major };
        // invert camelot number -> pitch class (inverse of camelot_for)
        let minor_inv: [usize; 13] = [0, 9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8];
        let major_inv: [usize; 13] = [0, 5, 6, 7, 8, 9, 10, 11, 0, 1, 2, 3, 4];
        let pc = match scale {
            Scale::Minor => minor_inv[n.min(12)],
            Scale::Major => major_inv[n.min(12)],
        };
        return (pc_to_name(pc).into(), format!("{n}{l}"), scale);
    }
    let upper = t.to_ascii_uppercase();
    let (root, scale) = if upper.ends_with("MINOR") {
        (&upper[..upper.len() - 5], Scale::Minor)
    } else if upper.ends_with("MAJOR") {
        (&upper[..upper.len() - 5], Scale::Major)
    } else if upper.ends_with("MIN") {
        (&upper[..upper.len() - 3], Scale::Minor)
    } else if upper.ends_with("MAJ") {
        (&upper[..upper.len() - 3], Scale::Major)
    } else if t.ends_with('m') {
        (&upper[..upper.len() - 1], Scale::Minor)
    } else {
        (upper.as_str(), Scale::Major)
    };
    let root = root.trim().replace('♭', "");
    let pc = NOTE_NAMES
        .iter()
        .position(|n| n.eq_ignore_ascii_case(root))
        .unwrap_or_else(|| flat_pc(root));
    (pc_to_name(pc).into(), camelot_for(pc, scale), scale)
}

fn flat_pc(root: &str) -> usize {
    let map = [("BB", 10), ("EB", 4), ("AB", 9), ("DB", 2), ("GB", 7), ("CB", 11)];
    for (k, v) in map {
        if root.eq_ignore_ascii_case(k) {
            return v;
        }
    }
    0
}

// ---------------------------------------------------------------------------
// Transition planning types (spec sections 2, 8)
// ---------------------------------------------------------------------------

/// The kind of automatic transition (spec section 2).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransitionType {
    /// Long beatmatched overlap with EQ automation and bass swap.
    SeamlessBeatmatchedBlend,
    /// Instant/short cut exactly on a downbeat/phrase boundary.
    QuickCut,
    /// Source decays through delay/reverb while target enters clean.
    EchoOut,
    /// HP/LP filter automation reduces spectral clash.
    FilterSweep,
    /// Explicit complementary low-band swap.
    BassSwap,
    /// Only drums/percussion overlap first, melodic layers later.
    DrumOnlyBridge,
    /// No PCM: plain equal-power crossfade using metadata only.
    MetadataFallbackCrossfade,
}

impl fmt::Display for TransitionType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            TransitionType::SeamlessBeatmatchedBlend => "seamless_beatmatched_blend",
            TransitionType::QuickCut => "quick_cut",
            TransitionType::EchoOut => "echo_out",
            TransitionType::FilterSweep => "filter_sweep",
            TransitionType::BassSwap => "bass_swap",
            TransitionType::DrumOnlyBridge => "drum_only_bridge",
            TransitionType::MetadataFallbackCrossfade => "metadata_fallback_crossfade",
        };
        write!(f, "{s}")
    }
}

/// Mixer parameter controlled by an automation lane.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MixParam {
    Volume,
    Low,
    Mid,
    High,
    Filter,
    Delay,
    Reverb,
}

impl fmt::Display for MixParam {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            MixParam::Volume => "volume",
            MixParam::Low => "low",
            MixParam::Mid => "mid",
            MixParam::High => "high",
            MixParam::Filter => "filter",
            MixParam::Delay => "delay",
            MixParam::Reverb => "reverb",
        };
        write!(f, "{s}")
    }
}

/// Role of a track within a transition.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrackRole {
    Source,
    Target,
}

/// Interpolation shape between automation points.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CurveShape {
    Linear,
    EqualPower,
    Expo,
    SCurve,
}

/// Control point of an automation curve.
#[derive(Clone, Copy, Debug)]
pub struct AutomationPoint {
    /// Time relative to transition start (seconds).
    pub time: f64,
    /// Parameter value 0..1 mapped to physical range by the mix engine.
    pub value: f64,
    pub curve: CurveShape,
}

/// A complete automation lane.
#[derive(Clone, Debug)]
pub struct AutomationCurve {
    pub parameter: MixParam,
    pub track_role: TrackRole,
    pub points: Vec<AutomationPoint>,
}

impl AutomationCurve {
    pub fn new(parameter: MixParam, track_role: TrackRole, points: Vec<AutomationPoint>) -> Self {
        Self { parameter, track_role, points }
    }

    /// Evaluate at time `t` relative to transition start.
    pub fn eval(&self, t: f64) -> f64 {
        if self.points.is_empty() {
            return 0.0;
        }
        if t <= self.points[0].time {
            return self.points[0].value;
        }
        for w in self.points.windows(2) {
            let (a, b) = (&w[0], &w[1]);
            if t >= a.time && t <= b.time {
                let x = if b.time > a.time { (t - a.time) / (b.time - a.time) } else { 1.0 };
                return interp(a.value, b.value, x, b.curve);
            }
        }
        self.points.last().unwrap().value
    }
}

/// Piecewise interpolation with curve shapes.
pub fn interp(a: f64, b: f64, x: f64, shape: CurveShape) -> f64 {
    let x = x.clamp(0.0, 1.0);
    let shaped = match shape {
        CurveShape::Linear => x,
        CurveShape::EqualPower => {
            if b >= a {
                x.sqrt()
            } else {
                1.0 - (1.0 - x).sqrt()
            }
        }
        CurveShape::Expo => x.powi(2),
        CurveShape::SCurve => x * x * (3.0 - 2.0 * x),
    };
    a + (b - a) * shaped
}

/// Complete plan for one transition (spec section 8).
#[derive(Clone, Debug)]
pub struct TransitionPlan {
    pub source_track_id: String,
    pub target_track_id: String,
    pub source_cue_time: f64,
    pub target_cue_time: f64,
    pub transition_type: TransitionType,
    /// Length of the overlapping transition region (seconds).
    pub duration: f64,
    pub beatmatch: bool,
    pub key_shift_semitones: f64,
    /// Playback rate for target after time stretch (1.0 = native tempo).
    pub tempo_ratio: f64,
    pub automation: Vec<AutomationCurve>,
    pub fallback_plan: Option<Box<TransitionPlan>>,
    pub confidence: f64,
    pub cost: TransitionCost,
    /// Expected silence gap (s) before target cue if grid forces a delay.
    pub expected_gap: f64,
}

/// Weighted cost terms (spec section 3).
#[derive(Clone, Copy, Debug, Default)]
pub struct TransitionCost {
    pub beat_alignment_error: f64,
    pub key_conflict_penalty: f64,
    pub phrase_misalignment_penalty: f64,
    pub bass_conflict_penalty: f64,
    pub vocal_collision_penalty: f64,
    pub energy_jump_penalty: f64,
    pub structural_mismatch_penalty: f64,
    pub artifact_penalty: f64,
    pub smoothness_bonus: f64,
    pub musical_interest_bonus: f64,
    /// Total weighted score (lower is better).
    pub total: f64,
}

/// Compatibility summary between two tracks.
#[derive(Clone, Debug)]
pub struct CompatibilityScore {
    pub overall: f64,
    pub harmonic: f64,
    pub rhythmic: f64,
    pub energy: f64,
    pub structural: f64,
    pub camelot_distance: usize,
    pub bpm_delta_abs: f64,
    pub bpm_delta_pct: f64,
}

/// Live playback state fed to the planner.
#[derive(Clone, Debug)]
pub struct PlaybackContext {
    pub current_time: f64,
    pub remaining_source_duration: f64,
    pub device: DeviceClass,
    pub force_now: bool,
}

impl Default for PlaybackContext {
    fn default() -> Self {
        Self {
            current_time: 0.0,
            remaining_source_duration: 60.0,
            device: DeviceClass::Desktop,
            force_now: false,
        }
    }
}

/// Device capability class for graceful degradation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceClass {
    Desktop,
    Mobile,
    /// Weak device: cheap DSP only (linear crossfade, no reverb/delay sends).
    LowPower,
}

/// User preference style for transitions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UserMixPreference {
    Smooth,
    Energetic,
    Invisible,
    Dramatic,
}

/// Hint about which queued track would mix in best next.
#[derive(Clone, Debug)]
pub struct TrackHint {
    pub track_id: String,
    pub suggested_transition: TransitionType,
    pub score: f64,
    pub reason: String,
}

/// Evaluation weights for the cost function (spec section 3).
#[derive(Clone, Copy, Debug)]
pub struct CostWeights {
    pub w1_beat_alignment: f64,
    pub w2_key_conflict: f64,
    pub w3_phrase_misalignment: f64,
    pub w4_bass_conflict: f64,
    pub w5_vocal_collision: f64,
    pub w6_energy_jump: f64,
    pub w7_structural_mismatch: f64,
    pub w8_artifact: f64,
    pub w9_smoothness_bonus: f64,
    pub w10_musical_interest_bonus: f64,
}

impl Default for CostWeights {
    fn default() -> Self {
        Self {
            w1_beat_alignment: 1.0,
            w2_key_conflict: 0.9,
            w3_phrase_misalignment: 0.7,
            w4_bass_conflict: 0.8,
            w5_vocal_collision: 0.6,
            w6_energy_jump: 0.5,
            w7_structural_mismatch: 0.45,
            w8_artifact: 0.35,
            w9_smoothness_bonus: 0.4,
            w10_musical_interest_bonus: 0.25,
        }
    }
}

impl CostWeights {
    /// Apply the weighted formula from spec section 3.
    pub fn evaluate(&self, c: &mut TransitionCost) {
        c.total = self.w1_beat_alignment * c.beat_alignment_error
            + self.w2_key_conflict * c.key_conflict_penalty
            + self.w3_phrase_misalignment * c.phrase_misalignment_penalty
            + self.w4_bass_conflict * c.bass_conflict_penalty
            + self.w5_vocal_collision * c.vocal_collision_penalty
            + self.w6_energy_jump * c.energy_jump_penalty
            + self.w7_structural_mismatch * c.structural_mismatch_penalty
            + self.w8_artifact * c.artifact_penalty
            - self.w9_smoothness_bonus * c.smoothness_bonus
            - self.w10_musical_interest_bonus * c.musical_interest_bonus;
    }

    /// Scale weights toward a user preference profile.
    pub fn for_preference(&self, p: UserMixPreference) -> Self {
        let mut w = *self;
        match p {
            UserMixPreference::Smooth => {
                w.w9_smoothness_bonus *= 1.6;
                w.w6_energy_jump *= 1.4;
                w.w10_musical_interest_bonus *= 0.6;
            }
            UserMixPreference::Energetic => {
                w.w10_musical_interest_bonus *= 1.8;
                w.w7_structural_mismatch *= 0.6;
                w.w9_smoothness_bonus *= 0.6;
            }
            UserMixPreference::Invisible => {
                w.w1_beat_alignment *= 1.5;
                w.w4_bass_conflict *= 1.5;
                w.w5_vocal_collision *= 1.4;
                w.w9_smoothness_bonus *= 1.8;
            }
            UserMixPreference::Dramatic => {
                w.w10_musical_interest_bonus *= 2.2;
                w.w3_phrase_misalignment *= 1.2;
            }
        }
        w
    }
}
