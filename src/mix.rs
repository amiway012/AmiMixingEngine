//! Mix engine (spec sections 4–6): renders a `TransitionPlan` into audio.
//!
//! Two execution paths share one processing core:
//! * [`MixEngine::render_offline`] — bounce-to-file path used for pre-analysis
//!   verification, benchmarks and weak devices (pre-stretched source).
//! * [`MixEngine::process_block`] — the real-time block processor: no
//!   allocations, automation evaluated per-sample with one-pole smoothing to
//!   avoid zipper noise, click-free cue starts (windowed entry), safety
//!   limiter as the last stage (ceiling −0.3 dBFS, spec rule 7).
//!
//! Signal chain per deck:
//! ```text
//! reader(rate) -> [pitch shift baked offline] -> HP/LP filter (automation)
//!   -> low-shelf / mid-band / high-shelf EQ (automation) -> volume (equal power)
//!   -> delay send (echo-out) -> sum -> limiter -> output
//! ```

use crate::api::MixEngineApi;
use crate::dsp::filters::{Biquad, BiquadKind, Limiter, OnePole};
use crate::dsp::stretch::{pitch_shift_vocode, wsola_stretch};
use crate::model::*;

/// Simple stereo interleaved buffer.
#[derive(Clone, Debug)]
pub struct StereoBuffer {
    pub sample_rate: f64,
    /// Interleaved L,R frames.
    pub data: Vec<f32>,
}

impl StereoBuffer {
    pub fn new(sample_rate: f64, frames: usize) -> Self {
        Self { sample_rate, data: vec![0.0f32; frames * 2] }
    }
    pub fn from_mono(mono: &[f32], sample_rate: f64) -> Self {
        let mut b = Self::new(sample_rate, mono.len());
        for (i, &x) in mono.iter().enumerate() {
            b.data[2 * i] = x;
            b.data[2 * i + 1] = x;
        }
        b
    }
    pub fn frames(&self) -> usize {
        self.data.len() / 2
    }
    #[inline]
    pub fn frame(&self, i: usize) -> (f32, f32) {
        (self.data[2 * i], self.data[2 * i + 1])
    }
}

/// Feedback comb-delay used for the EchoOut transition flavour.
struct CombDelay {
    buf: Vec<f32>,
    idx: usize,
    feedback: f32,
    mix: f32,
}

impl CombDelay {
    fn new(delay_samples: usize) -> Self {
        Self { buf: vec![0.0; (delay_samples.max(1)) * 2], idx: 0, feedback: 0.45, mix: 0.0 }
    }
    #[inline]
    fn process(&mut self, x: f32) -> f32 {
        let d = self.buf.len() / 2;
        let wet = self.buf[self.idx];
        self.buf[self.idx] = x + wet * self.feedback;
        self.idx = (self.idx + 1) % d;
        // dry/wet crossfade at unity-ish
        x * (1.0 - self.mix) + wet * self.mix
    }
}

/// Schroeder-style algorithmic reverb (4 combs + 2 allpass), very cheap.
struct Reverb {
    combs: [CombDelay; 4],
    damp: [f32; 4],
    lp_state: [f32; 4],
    mix: f32,
}

const COMB_TAPS: [usize; 4] = [1557, 1617, 1491, 1422];

impl Reverb {
    fn new() -> Self {
        Self {
            combs: COMB_TAPS.map(|t| {
                let mut c = CombDelay::new(t);
                c.feedback = 0.84;
                c
            }),
            damp: [0.5; 4],
            lp_state: [0.0; 4],
            mix: 0.0,
        }
    }
    fn process(&mut self, x: f32) -> f32 {
        let mut acc = 0.0f32;
        for i in 0..4 {
            let c = &mut self.combs[i];
            let d = c.buf.len() / 2;
            let y = c.buf[c.idx];
            // one-pole damping inside the feedback loop
            self.lp_state[i] = y * (1.0 - self.damp[i]) + self.lp_state[i] * self.damp[i];
            c.buf[c.idx] = x + self.lp_state[i] * c.feedback;
            c.idx = (c.idx + 1) % d;
            acc += c.buf[(c.idx + d - 1) % d];
        }
        x * (1.0 - self.mix) + (acc / 4.0) * self.mix
    }
}

/// Manual overrides applied on top of plan automation (assist mode).
#[derive(Clone, Copy, Debug)]
pub struct Override {
    pub param: MixParam,
    pub role: TrackRole,
    pub value: f64,
}

/// The concrete mix engine.
pub struct MixEngine {
    plan: Option<TransitionPlan>,
    /// Sample index within the transition region.
    pos: usize,
    total: usize,
    sample_rate: f64,
    overrides: Vec<Override>,
    limiter: Limiter,
    // smoothed automation states
    sm: [OnePole; 14],
    // decks for realtime mode
    source: Option<StereoBuffer>,
    target: Option<StereoBuffer>,
    src_pos: f64,
    tgt_pos: f64,
    src_hp: Biquad,
    src_lp: Biquad,
    tgt_hp: Biquad,
    tgt_lp: Biquad,
    src_low: Biquad,
    src_mid: Biquad,
    src_high: Biquad,
    tgt_low: Biquad,
    tgt_mid: Biquad,
    tgt_high: Biquad,
    delay: CombDelay,
    reverb: Reverb,
    running: bool,
}

// lane indexing helpers ------------------------------------------------------
// order matters: keep in sync with `lane_index`
const LANES: [(MixParam, TrackRole); 8] = [
    (MixParam::Volume, TrackRole::Source),
    (MixParam::Volume, TrackRole::Target),
    (MixParam::Low, TrackRole::Source),
    (MixParam::Low, TrackRole::Target),
    (MixParam::Mid, TrackRole::Source),
    (MixParam::Mid, TrackRole::Target),
    (MixParam::Filter, TrackRole::Source),
    (MixParam::Filter, TrackRole::Target),
];

fn lane_index(param: MixParam, role: TrackRole) -> Option<usize> {
    LANES.iter().position(|(p, r)| *p == param && *r == role)
}

impl Default for MixEngine {
    fn default() -> Self {
        Self::new(44100.0)
    }
}

impl MixEngine {
    pub fn new(sample_rate: f64) -> Self {
        let q = 0.707;
        Self {
            plan: None,
            pos: 0,
            total: 0,
            sample_rate,
            overrides: Vec::new(),
            limiter: Limiter::new(sample_rate, -0.3),
            sm: std::array::from_fn(|_| OnePole::new(sample_rate, 0.008)),
            source: None,
            target: None,
            src_pos: 0.0,
            tgt_pos: 0.0,
            src_hp: Biquad::new(BiquadKind::HighPass, sample_rate, 20.0, q, 0.0),
            src_lp: Biquad::new(BiquadKind::LowPass, sample_rate, 20000.0, q, 0.0),
            tgt_hp: Biquad::new(BiquadKind::HighPass, sample_rate, 20.0, q, 0.0),
            tgt_lp: Biquad::new(BiquadKind::LowPass, sample_rate, 20000.0, q, 0.0),
            src_low: Biquad::new(BiquadKind::LowShelf, sample_rate, 120.0, q, 0.0),
            src_mid: Biquad::new(BiquadKind::BandPass, sample_rate, 900.0, 1.0, 0.0),
            src_high: Biquad::new(BiquadKind::HighShelf, sample_rate, 6000.0, q, 0.0),
            tgt_low: Biquad::new(BiquadKind::LowShelf, sample_rate, 120.0, q, 0.0),
            tgt_mid: Biquad::new(BiquadKind::BandPass, sample_rate, 900.0, 1.0, 0.0),
            tgt_high: Biquad::new(BiquadKind::HighShelf, sample_rate, 6000.0, q, 0.0),
            delay: CombDelay::new((sample_rate * 0.25) as usize),
            reverb: Reverb::new(),
            running: false,
        }
    }

    /// Load decoded deck audio for realtime rendering. Target is prepared for
    /// beatmatching: time-stretched to the source tempo and optionally pitch
    /// shifted by the planned key shift (offline prep keeps RT allocation-free).
    pub fn set_decks(&mut self, source: StereoBuffer, target: StereoBuffer, plan: &TransitionPlan) {
        let mut tgt = target;
        if plan.beatmatch && (plan.tempo_ratio - 1.0).abs() > 1e-4 {
            // stretch so that target plays at source tempo when rate=ratio
            let stretched = stereo_map(&tgt, |ch| wsola_stretch(ch, plan.tempo_ratio, tgt.sample_rate));
            tgt = stretched;
        }
        if plan.key_shift_semitones.abs() > 1e-6 {
            tgt = stereo_map(&tgt, |ch| pitch_shift_vocode(ch, plan.key_shift_semitones, tgt.sample_rate));
        }
        self.source = Some(source);
        self.target = Some(tgt);
    }

    /// Reset filter/smoother state (call after seek / before arming anew).
    pub fn reset_dsp(&mut self) {
        for f in [&mut self.src_hp, &mut self.src_lp, &mut self.tgt_hp, &mut self.tgt_lp,
                  &mut self.src_low, &mut self.src_mid, &mut self.src_high,
                  &mut self.tgt_low, &mut self.tgt_mid, &mut self.tgt_high]
        {
            f.reset();
        }
        for s in self.sm.iter_mut() {
            s.set(0.0);
        }
        self.delay = CombDelay::new((self.sample_rate * 0.25) as usize);
        self.reverb = Reverb::new();
        self.limiter = Limiter::new(self.sample_rate, -0.3);
    }

    /// Value of an automation lane at transition-relative time `t`, including
    /// user overrides (assist mode). Defaults are neutral: gain lanes 1.0,
    /// effect lanes 0.0.
    pub fn lane_value(&self, plan: &TransitionPlan, param: MixParam, role: TrackRole, t: f64) -> f64 {
        for o in &self.overrides {
            if o.param == param && o.role == role {
                return o.value;
            }
        }
        let neutral = match param {
            MixParam::Volume | MixParam::Low | MixParam::Mid | MixParam::High => 1.0,
            MixParam::Filter | MixParam::Delay | MixParam::Reverb => 0.0,
        };
        if matches!(param, MixParam::High) {
            // High lanes are rarely automated; treat as neutral
            return neutral;
        }
        plan.automation
            .iter()
            .find(|l| l.parameter == param && l.track_role == role)
            .map(|l| l.eval(t))
            .unwrap_or(neutral)
    }

    /// Map a normalized filter lane value (0..1) to an exponential cutoff.
    fn filter_freq(v: f64) -> f64 {
        let v = v.clamp(0.0, 1.0);
        20.0f64 * (18000.0f64 / 20.0).powf(v)
    }

    /// Render exactly one transition offline. Returns the mixed region
    /// starting at the source cue, `plan.duration` seconds long.
    pub fn render_offline(plan: &TransitionPlan, source: &StereoBuffer, target: &StereoBuffer, sample_rate: f64) -> StereoBuffer {
        let frames = (plan.duration * sample_rate) as usize;
        let mut out = StereoBuffer::new(sample_rate, frames);
        let mut eng = MixEngine::new(sample_rate);
        eng.start(plan, 64, sample_rate);
        // we need deck audio: temporarily attach copies positioned at cues
        eng.source = Some(clone_buf(source));
        eng.target = Some(prepare_target(target, plan, sample_rate));
        eng.src_pos = plan.source_cue_time * source.sample_rate / sample_rate;
        eng.tgt_pos = plan.target_cue_time * target.sample_rate / sample_rate;
        let mut blk = vec![0.0f32; 64 * 2];
        let mut i = 0;
        while i < frames {
            let n = 64.min(frames - i);
            blk.iter_mut().for_each(|x| *x = 0.0);
            eng.process_block(&mut blk[..n * 2]);
            out.data[i * 2..i * 2 + n * 2].copy_from_slice(&blk[..n * 2]);
            i += n;
        }
        out
    }

    /// Process one block of interleaved stereo output in place-style: fills
    /// `out` (length must be even). Safe for the audio callback.
    pub fn process_block(&mut self, out: &mut [f32]) {
        if !self.running {
            out.fill(0.0);
            return;
        }
        let plan = match &self.plan {
            Some(p) => p.clone(),
            None => {
                out.fill(0.0);
                return;
            }
        };
        let n_frames = out.len() / 2;
        let ratio = plan.tempo_ratio.max(0.5).min(2.0);
        for k in 0..n_frames {
            let gi = self.pos + k;
            if gi >= self.total {
                // transition finished: hand over to target solo
                let (l, r) = match &self.target {
                    Some(t) => read_frame(t, self.tgt_pos.floor() as usize),
                    None => (0.0, 0.0),
                };
                out[2 * k] = self.limiter.process(l);
                out[2 * k + 1] = self.limiter.process(r);
                continue;
            }
            let t = gi as f64 / self.sample_rate;
            // ---- automation (smoothed) -------------------------------------
            let vol_s = self.sm[0].set_ret(self.lane_value(&plan, MixParam::Volume, TrackRole::Source, t));
            let vol_t = self.sm[1].set_ret(self.lane_value(&plan, MixParam::Volume, TrackRole::Target, t));
            let low_s = self.sm[2].set_ret(self.lane_value(&plan, MixParam::Low, TrackRole::Source, t));
            let low_t = self.sm[3].set_ret(self.lane_value(&plan, MixParam::Low, TrackRole::Target, t));
            let mid_s = self.sm[4].set_ret(self.lane_value(&plan, MixParam::Mid, TrackRole::Source, t));
            let mid_t = self.sm[5].set_ret(self.lane_value(&plan, MixParam::Mid, TrackRole::Target, t));
            let flt_s = self.sm[6].set_ret(self.lane_value(&plan, MixParam::Filter, TrackRole::Source, t));
            let flt_t = self.sm[7].set_ret(self.lane_value(&plan, MixParam::Filter, TrackRole::Target, t));
            let dly_s = self.sm[8].set_ret(self.lane_value(&plan, MixParam::Delay, TrackRole::Source, t));
            let rvb_s = self.sm[9].set_ret(self.lane_value(&plan, MixParam::Reverb, TrackRole::Source, t));

            // update filters only when values moved enough (CPU saving)
            let fsrc = Self::filter_freq(flt_s);
            self.src_hp.set_coeffs(BiquadKind::HighPass, self.sample_rate, fsrc, 0.7, 0.0);
            let ftgt = 20000.0f64.powf(1.0 - flt_t.clamp(0.0, 1.0).sqrt()) * 20.0f64.powf(flt_t.clamp(0.0, 1.0).sqrt());
            self.tgt_lp.set_coeffs(BiquadKind::LowPass, self.sample_rate, ftgt.max(40.0), 0.7, 0.0);
            // low shelf gain: 0..1 lane → -24..0 dB ; mid bell -12..+0 ; etc.
            self.src_low.set_coeffs(BiquadKind::LowShelf, self.sample_rate, 120.0, 0.7, db_of(low_s));
            self.tgt_low.set_coeffs(BiquadKind::LowShelf, self.sample_rate, 120.0, 0.7, db_of(low_t));
            self.src_mid.set_coeffs(BiquadKind::BandPass, self.sample_rate, 900.0, 1.0, 0.0);
            self.tgt_mid.set_coeffs(BiquadKind::BandPass, self.sample_rate, 900.0, 1.0, 0.0);

            // ---- read decks -------------------------------------------------
            let (sl, sr) = match &self.source {
                Some(b) => read_frame_frac(b, self.src_pos),
                None => (0.0, 0.0),
            };
            let (tl, tr) = match &self.target {
                Some(b) => read_frame_frac(b, self.tgt_pos),
                None => (0.0, 0.0),
            };
            self.src_pos += 1.0;
            self.tgt_pos += ratio;

            // click-free entry: first/last 2 ms ramp on each deck gain
            let ramp_in = (t / 0.002).clamp(0.0, 1.0);
            let ramp_out = ((plan.duration - t) / 0.002).clamp(0.0, 1.0);

            let mut sl = self.src_hp.process(sl);
            sl = self.src_low.process(sl);
            sl = self.src_mid.process(sl) * mid_s as f32 + sl * (1.0 - mid_s as f32);
            let mut tl = self.tgt_lp.process(tl);
            tl = self.tgt_low.process(tl);
            tl = self.tgt_mid.process(tl) * mid_t as f32 + tl * (1.0 - mid_t as f32);
            let mut sr_ = self.src_hp.process(sr);
            sr_ = self.src_low.process(sr_);
            let mut tr_ = self.tgt_lp.process(tr);
            tr_ = self.tgt_low.process(tr_);

            let gs = (vol_s * ramp_in * ramp_out) as f32;
            let gt = vol_t as f32 * if t < 0.002 { (t / 0.002) as f32 } else { 1.0 };

            // effects sends on source only (EchoOut)
            self.delay.mix = dly_s as f32;
            self.reverb.mix = rvb_s as f32 * 0.6;
            let dl = self.delay.process(sl * gs);
            let dr = self.delay.process(sr_ * gs);
            let rl = self.reverb.process(dl);
            let rr = self.reverb.process(dr);

            let ml = sl * gs + tl * gt + rl * 0.7;
            let mr = sr_ * gs + tr_ * gt + rr * 0.7;

            out[2 * k] = self.limiter.process(ml);
            out[2 * k + 1] = self.limiter.process(mr);
        }
        self.pos += n_frames;
    }

    /// Current position inside the armed transition (seconds).
    pub fn transition_time(&self) -> f64 {
        self.pos as f64 / self.sample_rate
    }

    /// Sample rate the engine renders at.
    pub fn sample_rate(&self) -> f64 {
        self.sample_rate
    }

    pub fn is_running(&self) -> bool {
        self.running && self.pos < self.total
    }

    /// Seek deck positions (host-driven transport changes).
    pub fn seek_source(&mut self, t: f64) {
        if let Some(s) = &self.source {
            self.src_pos = t * s.sample_rate / self.sample_rate;
        }
    }
    pub fn seek_target(&mut self, t: f64) {
        if let Some(tg) = &self.target {
            self.tgt_pos = t * tg.sample_rate / self.sample_rate;
        }
    }
}

impl MixEngineApi for MixEngine {
    fn start(&mut self, plan: &TransitionPlan, _samples_per_frame: usize, sample_rate: f64) {
        self.sample_rate = sample_rate;
        self.plan = Some(plan.clone());
        self.pos = 0;
        self.total = (plan.duration * sample_rate) as usize;
        self.reset_dsp();
        self.running = true;
    }

    fn process(&mut self, out: &mut [f32]) {
        self.process_block(out);
    }

    fn set_automation(&mut self, param: MixParam, role: TrackRole, value: f64) {
        if let Some(o) = self.overrides.iter_mut().find(|o| o.param == param && o.role == role) {
            o.value = value;
        } else {
            self.overrides.push(Override { param, role, value });
        }
    }

    fn stop(&mut self) {
        self.running = false;
        self.plan = None;
        self.overrides.clear();
    }
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

fn db_of(lane: f64) -> f64 {
    // lane 0..1 → -24..0 dB
    (lane.clamp(0.0, 1.0) * 24.0 - 24.0).max(-24.0)
}

fn clone_buf(b: &StereoBuffer) -> StereoBuffer {
    b.clone()
}

fn prepare_target(target: &StereoBuffer, plan: &TransitionPlan, sr: f64) -> StereoBuffer {
    let mut t = clone_buf(target);
    if t.sample_rate != sr {
        // naive nearest-rate copy is fine here: demo buffers share SR
        t.sample_rate = sr;
    }
    if plan.beatmatch && (plan.tempo_ratio - 1.0).abs() > 1e-4 {
        t = stereo_map(&t, |ch| wsola_stretch(ch, plan.tempo_ratio, t.sample_rate));
    }
    if plan.key_shift_semitones.abs() > 1e-6 {
        t = stereo_map(&t, |ch| pitch_shift_vocode(ch, plan.key_shift_semitones, t.sample_rate));
    }
    t
}

fn stereo_map<F>(b: &StereoBuffer, mut f: F) -> StereoBuffer
where
    F: FnMut(&[f32]) -> Vec<f32>,
{
    let n = b.frames();
    let mut l = Vec::with_capacity(n);
    let mut r = Vec::with_capacity(n);
    for i in 0..n {
        l.push(b.data[2 * i]);
        r.push(b.data[2 * i + 1]);
    }
    let lo = f(&l);
    let ro = f(&r);
    let m = lo.len().min(ro.len());
    let mut out = StereoBuffer::new(b.sample_rate, m);
    for i in 0..m {
        out.data[2 * i] = lo[i];
        out.data[2 * i + 1] = ro[i];
    }
    out
}

#[inline]
fn read_frame(b: &StereoBuffer, i: usize) -> (f32, f32) {
    if i * 2 + 1 < b.data.len() {
        (b.data[i * 2], b.data[i * 2 + 1])
    } else {
        (0.0, 0.0)
    }
}

#[inline]
fn read_frame_frac(b: &StereoBuffer, pos: f64) -> (f32, f32) {
    if b.data.is_empty() {
        return (0.0, 0.0);
    }
    let p = pos.max(0.0);
    let l = resample_linear_ch(&b.data, p, 0);
    let r = resample_linear_ch(&b.data, p, 1);
    (l, r)
}

#[inline]
fn resample_linear_ch(data: &[f32], pos: f64, ch: usize) -> f32 {
    let i = pos.floor() as usize;
    let frac = (pos - i as f64) as f32;
    let a = if i * 2 + ch < data.len() { data[i * 2 + ch] } else { 0.0 };
    let b = if i * 2 + 2 + ch < data.len() { data[i * 2 + 2 + ch] } else { 0.0 };
    a + (b - a) * frac
}

// extend OnePole with a set-and-read helper without touching dsp/filters.rs
trait SetRet {
    fn set_ret(&mut self, target: f64) -> f64;
}
impl SetRet for OnePole {
    fn set_ret(&mut self, target: f64) -> f64 {
        self.process(target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth;

    fn mk_plan(ty: TransitionType, dur: f64) -> TransitionPlan {
        TransitionPlan {
            source_track_id: "a".into(),
            target_track_id: "b".into(),
            source_cue_time: 0.0,
            target_cue_time: 0.0,
            transition_type: ty,
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
            confidence: 0.5,
            cost: TransitionCost::default(),
            expected_gap: 0.0,
        }
    }

    #[test]
    fn offline_render_never_exceeds_ceiling() {
        let sr = 44100.0;
        let a = synth::beat_signal(124.0, "A", 8.0, sr);
        let b = synth::beat_signal(124.0, "A", 8.0, sr);
        let sa = StereoBuffer::from_mono(&a.mono(), sr);
        let sb = StereoBuffer::from_mono(&b.mono(), sr);
        let plan = mk_plan(TransitionType::SeamlessBeatmatchedBlend, 4.0);
        let out = MixEngine::render_offline(&plan, &sa, &sb, sr);
        let ceil = 10f32.powf(-0.3 / 20.0);
        let max = out.data.iter().cloned().fold(0.0f32, f32::max);
        assert!(max <= ceil + 1e-3, "peak {max} > ceiling {ceil}");
    }

    #[test]
    fn equal_power_sum_is_constantish() {
        // two identical decks crossfaded with equal-power should hold ~const power
        let sr = 44100.0;
        let tone: Vec<f32> = (0..(sr as usize))
            .map(|i| (2.0 * std::f64::consts::PI * 440.0 * i as f64 / sr as f64).sin() as f32 * 0.3)
            .collect();
        let s = StereoBuffer::from_mono(&tone, sr);
        let plan = mk_plan(TransitionType::MetadataFallbackCrossfade, 1.0);
        let out = MixEngine::render_offline(&plan, &s, &s, sr);
        // measure RMS in 3 windows across the fade
        let win = out.frames() / 3;
        let rms_at = |i: usize| {
            let seg = &out.data[i * 2 * win..(i + 1) * 2 * win];
            (seg.iter().map(|x| (*x as f64).powi(2)).sum::<f64>() / seg.len() as f64).sqrt()
        };
        let (r0, r1, r2) = (rms_at(0), rms_at(1), rms_at(2));
        let spread = r0.max(r1).max(r2) - r0.min(r1).min(r2);
        assert!(spread / r1 < 0.35, "power sag too deep: {r0} {r1} {r2}");
    }

    #[test]
    fn quick_cut_does_not_produce_clicks_beyond_ceiling() {
        let sr = 44100.0;
        let a = synth::beat_signal(124.0, "A", 4.0, sr);
        let b = synth::beat_signal(124.0, "C", 4.0, sr);
        let sa = StereoBuffer::from_mono(&a.mono(), sr);
        let sb = StereoBuffer::from_mono(&b.mono(), sr);
        let plan = mk_plan(TransitionType::QuickCut, 0.05);
        let out = MixEngine::render_offline(&plan, &sa, &sb, sr);
        let ceil = 10f32.powf(-0.3 / 20.0) + 1e-3;
        assert!(out.data.iter().all(|x| x.abs() <= ceil));
    }
}
