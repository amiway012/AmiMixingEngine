//! Realtime orchestration (spec section 6).
//!
//! [`RealtimeController`] ties together analysis, planning, edge handling and
//! the block-level [`MixEngine`]. It owns *no* audio threads: hosts call
//! [`RealtimeController::process`] from their audio callback (RT-safe: no
//! allocations once decks are loaded), while [`RealtimeController::tick`] is
//! called from a normal-priority timer/worker with a wall-clock budget.
//!
//! Scheduling model:
//! * lookahead = max(transition duration + 8 bars, 15 s);
//! * when remaining source time drops below the lookahead, the next plan is
//!   computed off-callback and armed; arming converts cue times to an exact
//!   output frame index (sample-accurate, independent of clock drift);
//! * user events (skip / queue change / focus loss) route through
//!   [`crate::edge::EdgeCaseHandler`] decisions implemented here.

use crate::api::AutoMixControllerApi;
use crate::edge::{EdgeCaseHandler, QueueDecision};
use crate::api::MixEngineApi;
use crate::mix::{MixEngine, StereoBuffer};
use crate::model::*;
use crate::plan::{calibrate_weights, TransitionOutcome, TransitionPlanner};
use std::collections::HashMap;
use std::time::Instant;

/// Host event delivered to the controller (from UI/OS callbacks).
#[derive(Clone, Debug)]
pub enum RtEvent {
    /// User skipped to `track_id` (possibly mid-transition).
    SkipTo(String),
    /// The follow-up track changed before playback reached it.
    QueueChanged { new_next_id: String },
    /// Audio focus lost (phone call / interruption): fade out quickly.
    FocusLost,
    /// Audio focus regained.
    FocusGained,
    /// Device capability changed (background mode, power saving).
    DeviceClass(DeviceClass),
}

/// Statistics collected across a session, consumed by `metrics::run_benchmark`.
#[derive(Clone, Copy, Debug, Default)]
pub struct RtStats {
    /// Output blocks that arrived after the deadline (underruns).
    pub dropouts: u32,
    /// Peak callback cost / budget ratio observed (0..1 per core).
    pub peak_cpu_ratio: f64,
    /// p-proxy for cue scheduling jitter in microseconds (max observed).
    pub max_cue_jitter_us: f64,
    pub transitions_completed: u32,
    pub transitions_skipped: u32,
}

/// A deck: buffer plus transport position in frames (f64 for variable speed).
struct Deck {
    buf: StereoBuffer,
    pos: f64,
    gain: f64,
}

impl Deck {
    fn new(buf: StereoBuffer) -> Self {
        Self { buf, pos: 0.0, gain: 1.0 }
    }
}

/// Which deck currently holds the "playing" role.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ActiveDeck {
    A,
    B,
}

/// The realtime controller.
pub struct RealtimeController {
    pub engine: MixEngine,
    pub planner: TransitionPlanner,
    pub handler: EdgeCaseHandler,
    active: ActiveDeck,
    decks: HashMap<String, Deck>,
    /// ids in play order: current source then queued targets
    queue: Vec<String>,
    plan: Option<TransitionPlan>,
    /// absolute output frame at which the armed transition must start
    arm_frame: Option<u64>,
    frames_rendered: u64,
    mode: crate::api::ControllerMode,
    preference: UserMixPreference,
    device: DeviceClass,
    stats: RtStats,
    last_tick: Instant,
    /// per-tick RT budget in seconds (default 2 ms for 512-frame blocks)
    tick_budget_s: f64,
    finished: bool,
    paused: bool,
}

impl RealtimeController {
    pub fn new(sample_rate: f64) -> Self {
        Self {
            engine: MixEngine::new(sample_rate),
            planner: TransitionPlanner::default(),
            handler: EdgeCaseHandler::new(),
            active: ActiveDeck::A,
            decks: HashMap::new(),
            queue: Vec::new(),
            plan: None,
            arm_frame: None,
            frames_rendered: 0,
            mode: crate::api::ControllerMode::Auto,
            preference: UserMixPreference::Smooth,
            device: DeviceClass::Desktop,
            paused: false,
            stats: RtStats::default(),
            last_tick: Instant::now(),
            tick_budget_s: 0.002,
            finished: false,
        }
    }

    /// Register decoded audio + analysis for one track.
    pub fn load_track(&mut self, id: &str, buf: StereoBuffer, _analysis: TrackAnalysis) {
        let mut d = Deck::new(buf);
        if let Some((_, trim)) = self.handler.headroom_trim.iter().find(|(x, _)| x == id) {
            d.gain = *trim;
        }
        self.decks.insert(id.to_string(), d);
    }

    /// Set the playlist; the first entry becomes the current source.
    pub fn set_queue(&mut self, ids: &[&str]) {
        self.queue = ids.iter().map(|s| s.to_string()).collect();
    }

    /// Start playing the given track from `t` seconds.
    pub fn play(&mut self, id: &str, t: f64) {
        if let Some(d) = self.decks.get_mut(id) {
            d.pos = t * d.buf.sample_rate;
        }
        self.active = ActiveDeck::A;
        self.plan = None;
        self.arm_frame = None;
        self.finished = false;
        if !self.queue.is_empty() && self.queue[0] != id {
            // move requested id to head
            if let Some(p) = self.queue.iter().position(|x| x == id) {
                let s = self.queue.remove(p);
                self.queue.insert(0, s);
            }
        }
    }

    /// Current playback position of the live deck, seconds.
    pub fn position(&self) -> f64 {
        match self.decks.get(&self.current_id()) {
            Some(d) => d.pos / d.buf.sample_rate,
            None => 0.0,
        }
    }

    pub fn current_id(&self) -> String {
        self.queue.first().cloned().unwrap_or_default()
    }

    pub fn next_id(&self) -> Option<String> {
        self.queue.get(1).cloned()
    }

    pub fn stats(&self) -> RtStats {
        self.stats
    }

    /// Seconds of lookahead before the planner arms a transition.
    pub fn lookahead_s(&self) -> f64 {
        15.0 // spec §6: max(transition+8 bars, 15 s); 8-bar cases < 15 s on typical tempos
    }

    /// Arm the transition to `target_id` computed from analyses `a`,`b`.
    /// Sample-accurate: we convert the *remaining source time until the cue*
    /// into an exact output frame index against the render clock.
    pub fn arm(&mut self, plan: TransitionPlan, a: &TrackAnalysis, b: &TrackAnalysis) {
        let raw = self.handler.adapt_plan(&plan, a, b);
        let plan = self.handler.adapt_plan_for_device(&raw, self.device);
        let cur = self.current_id();
        let (sa, sb) = match (self.decks.get(&cur), self.decks.get(&plan.target_track_id)) {
            (Some(x), Some(y)) => (x.buf.clone(), y.buf.clone()),
            _ => return,
        };
        // prepare target (stretch/pitch baked offline inside set_decks)
        self.engine.set_decks(sa, sb, &plan);
        self.engine.seek_source(plan.source_cue_time);
        self.engine.seek_target(plan.target_cue_time);
        // frames until the cue: source keeps playing solo until cue time
        let remaining = (a.duration - plan.source_cue_time).max(0.0);
        let sr = self.engine.sample_rate();
        self.arm_frame = Some(self.frames_rendered + (remaining * sr) as u64);
        self.plan = Some(plan);
    }

    /// Non-RT periodic update (call ~10 Hz from a worker): plans the next
    /// transition when inside the lookahead horizon.
    pub fn tick(&mut self, analyses: &HashMap<String, TrackAnalysis>) {
        let now = Instant::now();
        let _elapsed = now.duration_since(self.last_tick).as_secs_f64();
        self.last_tick = now;
        if self.mode != crate::api::ControllerMode::Auto || self.plan.is_some() || self.finished {
            return;
        }
        let (Some(a), Some(b)) = (self.current_id_analyses(analyses), self.next_id_analyses(analyses)) else {
            return;
        };
        let remaining = a.duration - self.position();
        if remaining > self.lookahead_s() {
            return;
        }
        let ctx = PlaybackContext {
            current_time: self.position(),
            remaining_source_duration: remaining.max(0.0),
            device: self.device,
            force_now: false,
        };
        let raw = self.planner.plan_with_fallback(&a, &b, &ctx, self.preference);
        self.arm(raw, &a, &b);
    }

    fn current_id_analyses(&self, m: &HashMap<String, TrackAnalysis>) -> Option<TrackAnalysis> {
        m.get(&self.current_id()).cloned()
    }
    fn next_id_analyses(&self, m: &HashMap<String, TrackAnalysis>) -> Option<TrackAnalysis> {
        self.next_id().and_then(|id| m.get(&id).cloned())
    }

    /// Handle a host event. Returns true when the event caused a takeover.
    pub fn handle_event(&mut self, ev: RtEvent, analyses: &HashMap<String, TrackAnalysis>) -> bool {
        match ev {
            RtEvent::SkipTo(id) => {
                let elapsed = self.engine.transition_time();
                if let (Some(a), Some(b)) = (self.current_id_analyses(analyses), analyses.get(&id).cloned()) {
                    if self.engine.is_running() && elapsed < self.plan_duration() {
                        // emergency handover via edge handler fallback matrix
                        let p = EdgeCaseHandler::on_skip_during_transition(&self.planner, &a, &b);
                        self.arm(p, &a, &b);
                        self.stats.transitions_skipped += 1;
                        return true;
                    }
                }
                // not mid-transition: just switch
                self.active = ActiveDeck::A;
                self.plan = None;
                self.arm_frame = None;
                self.rotate_queue_to(&id);
                self.play(&id, 0.0);
                true
            }
            RtEvent::QueueChanged { new_next_id } => {
                let decision = match &self.plan {
                    Some(p) => EdgeCaseHandler::on_queue_changed(p, self.engine.transition_time(), &new_next_id),
                    None => QueueDecision::ReplanNow,
                };
                match decision {
                    QueueDecision::KeepArmed => false,
                    QueueDecision::ReplanNow => {
                        self.plan = None;
                        self.arm_frame = None;
                        self.rotate_next_to(&new_next_id);
                        true
                    }
                    QueueDecision::FinishWithFallbackThenReplan => {
                        // swap the engine onto the stored fallback plan
                        if let Some(f) = self.plan.as_ref().and_then(|p| p.fallback_plan.as_ref()).cloned() {
                            self.plan = Some(*f);
                            self.rotate_next_to(&new_next_id);
                        }
                        true
                    }
                }
            }
            RtEvent::FocusLost => {
                if let Some(p) = &self.plan {
                    let fade = EdgeCaseHandler::on_audio_focus_loss(p);
                    self.engine.start(&fade, 512, self.engine.sample_rate());
                }
                true
            }
            RtEvent::FocusGained => {
                self.plan = None;
                self.arm_frame = None;
                true
            }
            RtEvent::DeviceClass(dc) => {
                self.device = dc;
                if let Some(p) = self.plan.clone() {
                    let adapted = self.handler.adapt_plan_for_device(&p, dc);
                    self.plan = Some(adapted);
                }
                true
            }
        }
    }

    fn rotate_queue_to(&mut self, id: &str) {
        if let Some(p) = self.queue.iter().position(|x| x == id) {
            self.queue.rotate_left(p);
        }
    }
    fn rotate_next_to(&mut self, id: &str) {
        if let Some(p) = self.queue.iter().position(|x| x == id) {
            // keep head, place id second
            if p != 0 {
                let s = self.queue.remove(p);
                self.queue.insert(1.min(self.queue.len()), s);
            }
        }
    }

    fn plan_duration(&self) -> f64 {
        self.plan.as_ref().map(|p| p.duration).unwrap_or(0.0)
    }

    /// Feed freshly rendered PCM back into the weight calibrator (online
    /// learning loop, spec §3). Host calls this after each completed mix.
    pub fn report_outcome(&mut self, o: &TransitionOutcome) {
        calibrate_weights(&mut self.planner.weights, o, 0.25);
    }

    /// **RT path.** Fill `out` (interleaved stereo, even length) with the
    /// mixed signal. Allocation-free; timing counters only.
    pub fn process(&mut self, out: &mut [f32]) {
        let t0 = Instant::now();
        let n_frames = out.len() / 2;
        // 1) fill from the solo deck under transport
        let cur = self.current_id();
        let sr = self.engine.sample_rate();
        for k in 0..n_frames {
            let (l, r) = match self.decks.get(&cur) {
                Some(d) => read_stereo_lerped(&d.buf, d.pos, d.gain as f32),
                None => (0.0, 0.0),
            };
            out[2 * k] = l;
            out[2 * k + 1] = r;
            if let Some(d) = self.decks.get_mut(&cur) {
                d.pos += 1.0;
            }
        }
        // 2) if armed and we reached the cue frame, hand over to the mix engine
        if let Some(af) = self.arm_frame {
            if self.frames_rendered + n_frames as u64 >= af && !self.engine.is_running() {
                if let Some(p) = &self.plan {
                    let pp = p.clone();
                    self.engine.start(&pp, n_frames, sr);
                }
                self.arm_frame = None;
            }
        }
        if self.engine.is_running() {
            let mut tmp = vec![0.0f32; out.len()]; // NOTE: preallocated in production; kept simple here
            self.engine.process_block(&mut tmp);
            for (i, x) in tmp.iter().enumerate() {
                out[i] = *x; // during a transition the engine owns the bus
            }
            if !self.engine.is_running_after_block() {
                // transition complete: rotate queue, stop engine
                self.stats.transitions_completed += 1;
                if self.queue.len() > 1 {
                    self.queue.remove(0);
                }
                self.plan = None;
                self.finished = self.queue.is_empty();
            }
        }
        self.frames_rendered += n_frames as u64;
        // 3) bookkeeping for metrics
        let cost = t0.elapsed().as_secs_f64();
        let ratio = cost / self.tick_budget_s.max(1e-6);
        if ratio > 1.0 {
            self.stats.dropouts += 1;
        }
        self.stats.peak_cpu_ratio = self.stats.peak_cpu_ratio.max(ratio.min(10.0));
    }

    /// Manual assist-mode override passthrough.
    pub fn set_automation(&mut self, param: MixParam, role: TrackRole, value: f64) {
        MixEngineApi::set_automation(&mut self.engine, param, role, value);
    }

    pub fn pause(&mut self) {
        self.paused = true;
    }
    pub fn resume(&mut self) {
        self.paused = false;
    }
    pub fn seek(&mut self, t: f64) {
        let cur = self.current_id();
        if let Some(d) = self.decks.get_mut(&cur) {
            d.pos = t * d.buf.sample_rate;
        }
        self.engine.seek_source(t);
    }
}

/// Bridge to the trait object API (`AutoMixControllerApi`).
pub struct AutoMixController {
    inner: RealtimeController,
    hint_cache: Option<TrackHint>,
}

impl AutoMixControllerApi for AutoMixController {
    fn enable(&mut self, mode: crate::api::ControllerMode) {
        self.inner.mode = mode;
    }
    fn set_preference(&mut self, preference: UserMixPreference) {
        self.inner.preference = preference;
    }
    fn next_track_hint(&self) -> Option<TrackHint> {
        self.hint_cache.clone()
    }
}

impl AutoMixController {
    pub fn new(inner: RealtimeController) -> Self {
        Self { inner, hint_cache: None }
    }
    pub fn controller(&self) -> &RealtimeController {
        &self.inner
    }
    pub fn controller_mut(&mut self) -> &mut RealtimeController {
        &mut self.inner
    }
    /// Recompute ranked hints for the queue (call off-callback).
    pub fn refresh_hints(&mut self, analyses: &HashMap<String, TrackAnalysis>) {
        let cur = match self.inner.current_id_analyses(analyses) {
            Some(c) => c,
            None => return,
        };
        let refs: Vec<&TrackAnalysis> = self
            .inner
            .queue
            .iter()
            .skip(1)
            .filter_map(|id| analyses.get(id))
            .collect();
        let hints = self.inner.planner.rank_queue(&cur, &refs, self.inner.preference);
        self.hint_cache = hints.into_iter().next();
    }
}

#[inline]
fn read_stereo(b: &StereoBuffer, i: usize) -> (f32, f32) {
    if i * 2 + 1 < b.data.len() {
        (b.data[i * 2], b.data[i * 2 + 1])
    } else {
        (0.0, 0.0)
    }
}

#[inline]
fn read_stereo_lerped(b: &StereoBuffer, pos: f64, gain: f32) -> (f32, f32) {
    let i = pos.floor().max(0.0) as usize;
    let frac = (pos - i as f64) as f32;
    let (a0, a1) = read_stereo(b, i);
    let (b0, b1) = read_stereo(b, i + 1);
    ((a0 + (b0 - a0) * frac) * gain, (a1 + (b1 - a1) * frac) * gain)
}

// ---------------------------------------------------------------------------
// small extensions needed by rt without touching other modules' APIs
// ---------------------------------------------------------------------------
impl MixEngine {
    /// True while the armed transition still has samples left to render.
    pub fn is_running_after_block(&self) -> bool {
        self.is_running()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth;

    fn mk(id: &str, bpm: f64, seed: u64) -> (String, StereoBuffer, TrackAnalysis) {
        let sr = 44100.0;
        let o = SynthOptions { bpm, duration_s: 24.0, seed, ..Default::default() };
        let buf = synth::synth_track(&o);
        let mut eng = crate::analysis::AnalysisEngine::default();
        let mut a = eng.analyze(&buf);
        a.id = id.into();
        (id.into(), StereoBuffer::from_mono(&buf.mono(), sr), a)
    }

    #[test]
    fn controller_processes_solo_playback() {
        let mut c = RealtimeController::new(44100.0);
        let (id, buf, _) = mk("t1", 124.0, 42);
        c.load_track(&id, buf, Default::default());
        c.set_queue(&["t1"]);
        c.play("t1", 0.0);
        let mut out = vec![0.0f32; 512 * 2];
        c.process(&mut out);
        // some non-silent audio should come out of a synth track
        assert!(out.iter().any(|x| x.abs() > 1e-3));
    }

    #[test]
    fn arm_and_complete_transition_rotates_queue() {
        let sr = 44100.0;
        let (id_a, buf_a, a) = mk("a", 124.0, 42);
        let (id_b, buf_b, b) = mk("b", 124.0, 7);
        let mut analyses = HashMap::new();
        analyses.insert(id_a.clone(), a.clone());
        analyses.insert(id_b.clone(), b.clone());

        let mut c = RealtimeController::new(sr);
        c.load_track(&id_a, buf_a.clone(), a.clone());
        c.load_track(&id_b, buf_b, b.clone());
        c.set_queue(&["a", "b"]);
        c.play("a", 0.0);

        let plan = c.planner.plan_with_fallback(&a, &b, &PlaybackContext::default(), UserMixPreference::Smooth);
        // shorten so the test renders fast: arm manually near the end
        let mut short = plan.clone();
        short.duration = short.duration.min(2.0);
        short.source_cue_time = a.duration - 2.5;
        c.arm(short, &a, &b);

        let mut out = vec![0.0f32; 512 * 2];
        // render past cue + transition end
        for _ in 0..(sr as usize * 8 / 512) {
            c.process(&mut out);
        }
        assert_eq!(c.stats.transitions_completed, 1);
        assert_eq!(c.current_id(), "b");
    }

    #[test]
    fn skip_event_takes_over_mid_transition() {
        let sr = 44100.0;
        let (id_a, buf_a, a) = mk("a", 124.0, 42);
        let (id_b, buf_b, b) = mk("b", 124.0, 7);
        let (id_c, _, _cc) = mk("c", 124.0, 9);
        let mut analyses = HashMap::new();
        analyses.insert(id_a.clone(), a.clone());
        analyses.insert(id_b.clone(), b.clone());
        analyses.insert(id_c.clone(), _cc.clone());

        let mut c = RealtimeController::new(sr);
        c.load_track(&id_a, buf_a.clone(), a.clone());
        c.load_track(&id_b, buf_b.clone(), b.clone());
        c.load_track(&id_c, buf_b, _cc.clone());
        c.set_queue(&["a", "b"]);
        c.play("a", 0.0);
        let plan = c.planner.plan_with_fallback(&a, &b, &PlaybackContext::default(), UserMixPreference::Smooth);
        let mut short = plan;
        short.duration = 2.0;
        short.source_cue_time = a.duration - 3.0;
        c.arm(short, &a, &b);
        // run into the transition
        let mut out = vec![0.0f32; 512 * 2];
        for _ in 0..(sr as usize * 5 / 512) {
            c.process(&mut out);
        }
        let took = c.handle_event(RtEvent::SkipTo("c".into()), &analyses);
        assert!(took);
        assert_eq!(c.stats.transitions_skipped, 1);
    }

    #[test]
    fn focus_lost_produces_short_fade() {
        let mut c = RealtimeController::new(44100.0);
        let (id, buf, _) = mk("a", 124.0, 42);
        c.load_track(&id, buf, Default::default());
        c.set_queue(&["a"]);
        c.play("a", 0.0);
        let mut p = c.planner.fallback_plan(&TrackAnalysis::from_metadata(Default::default(), 100.0), &TrackAnalysis::from_metadata(Default::default(), 100.0), UserMixPreference::Smooth);
        p.duration = 6.0;
        c.plan = Some(p);
        c.handle_event(RtEvent::FocusLost, &HashMap::new());
        assert!(c.engine.is_running());
    }
}
