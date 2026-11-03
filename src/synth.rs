//! Deterministic synthetic signal generator used for tests, demos and the
//! benchmark dataset (spec section 10). No external decoder dependency is
//! needed to exercise the full pipeline.

use crate::model::{AudioBuffer, NOTE_NAMES};

/// Small xorshift RNG for reproducibility.
pub struct Rng(pub u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x9E3779B97F4A7C15))
    }
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    /// Uniform in [-1, 1].
    #[inline]
    pub fn f(&mut self) -> f64 {
        (self.next_u64() as f64 / u64::MAX as f64) * 2.0 - 1.0
    }
}

/// Pitch class of a note name ("A", "F#", "Bb"…).
pub fn name_to_pc(name: &str) -> usize {
    let up = name.to_ascii_uppercase().replace('♯', "#").replace("#", "#");
    if let Some(i) = NOTE_NAMES.iter().position(|n| *n == up) {
        return i;
    }
    let flats = [("BB", 10), ("EB", 4), ("AB", 9), ("DB", 2), ("GB", 7), ("CB", 11)];
    for (k, v) in flats {
        if up == k {
            return v;
        }
    }
    0
}

/// Frequency of pitch class `pc` in octave `oct` (A4 = 440).
pub fn pc_freq(pc: usize, oct: i32) -> f64 {
    let midi = (oct + 1) * 12 + pc as i32; // C4 = midi 60 when oct=4? use A4=440 anchor
    440.0 * 2.0f64.powf((midi - 69) as f64 / 12.0)
}

/// Kick drum synth: exponentially decaying sine sweep 120→45 Hz + click.
fn kick(rng: &mut Rng, t: f64, fs: f64) -> f32 {
    if t < 0.0 || t > 0.35 {
        return 0.0;
    }
    let f = 45.0 + 75.0 * (-t / 0.05).exp();
    let env = (-t * 9.0).exp();
    let body = (2.0 * std::f64::consts::PI * f * t).sin() * env;
    let click = if t < 0.004 { rng.f() * (1.0 - t / 0.004) * 0.6 } else { 0.0 };
    (body * 0.9 + click) as f32
}

/// Snare: band-limited noise burst + 180 Hz tone.
fn snare(rng: &mut Rng, t: f64, _fs: f64) -> f32 {
    if t < 0.0 || t > 0.25 {
        return 0.0;
    }
    let env = (-t * 18.0).exp();
    let n = rng.f() * env * 0.7;
    let tone = (2.0 * std::f64::consts::PI * 180.0 * t).sin() * env * 0.4;
    (n + tone) as f32
}

/// Hi-hat: short high-passed noise.
fn hat(rng: &mut Rng, t: f64, dur: f64) -> f32 {
    if t < 0.0 || t > dur {
        return 0.0;
    }
    let env = (-t * 60.0).exp();
    (rng.f() * env * 0.25) as f32
}

/// Chord/pad synth for harmonic content. `pcs` are pitch classes.
fn chord_wave(t: f64, pcs: &[usize], oct: i32, amp: f64) -> f64 {
    let mut s = 0.0;
    for &pc in pcs {
        let f = pc_freq(pc, oct);
        // slight detune per voice
        s += (2.0 * std::f64::consts::PI * f * t).sin() * 0.5;
        s += (2.0 * std::f64::consts::PI * f * 1.003 * t).sin() * 0.25;
    }
    s * amp / (pcs.len().max(1) as f64)
}

/// Major/minor triad pitch classes from root pc.
pub fn triad(root_pc: usize, minor: bool) -> Vec<usize> {
    if minor {
        vec![root_pc, (root_pc + 3) % 12, (root_pc + 7) % 12]
    } else {
        vec![root_pc, (root_pc + 4) % 12, (root_pc + 7) % 12]
    }
}

/// Options controlling the synthesized "track".
#[derive(Clone, Debug)]
pub struct SynthOptions {
    pub bpm: f64,
    /// Root pitch class of the key.
    pub key_pc: usize,
    pub minor: bool,
    pub duration_s: f64,
    pub sample_rate: f64,
    /// Add a vocal-like formant melody line (for vocal collision testing).
    pub vocals: bool,
    /// Loudness scaling (linear gain).
    pub gain: f64,
    /// If true, add strong sustained bass on every beat (bass-conflict material).
    pub heavy_bass: bool,
    pub seed: u64,
}

impl Default for SynthOptions {
    fn default() -> Self {
        Self {
            bpm: 124.0,
            key_pc: name_to_pc("A"),
            minor: true,
            duration_s: 32.0,
            sample_rate: 44100.0,
            vocals: false,
            gain: 0.5,
            heavy_bass: true,
            seed: 42,
        }
    }
}

/// Generate a full DJ-ish track: 4/4 kick, hats on 8ths, offbeat bass, chord
/// pad changing every 4 bars along a simple progression, optional vocal line.
/// Structure: intro (drums only, first 8 bars), main, outro fade last 4 bars.
pub fn synth_track(o: &SynthOptions) -> AudioBuffer {
    let fs = o.sample_rate;
    let n = (o.duration_s * fs) as usize;
    let spb = 60.0 / o.bpm; // seconds per beat
    let bar = spb * 4.0;
    let mut rng = Rng::new(o.seed);
    let mut buf = vec![0.0f32; n];
    let root = o.key_pc;
    // simple progression degrees: i - VI - VII - v (minor) or I - vi - IV - V
    let prog: Vec<usize> = if o.minor {
        vec![root, (root + 8) % 12, (root + 10) % 12, (root + 7) % 12]
    } else {
        vec![root, (root + 9) % 12, (root + 5) % 12, (root + 7) % 12]
    };
    for i in 0..n {
        let t = i as f64 / fs;
        let beat_f = t / spb;
        let beat_idx = beat_f.floor() as i64;
        let beat_phase = beat_f - beat_f.floor();
        let bar_idx = (t / bar).floor() as i64;
        let intro = t < bar * 2.0;
        let outro_start = o.duration_s - bar * 2.0;
        let mut s = 0.0f64;
        // drums (skip most of intro: intro = hats only)
        if !intro {
            let kt = beat_phase * spb;
            s += kick(&mut rng, kt, fs) as f64;
            // snare on beats 2 & 4
            if beat_idx % 4 == 1 || beat_idx % 4 == 3 {
                s += snare(&mut rng, kt, fs) as f64 * 0.8;
            }
            // hats on 8ths
            s += hat(&mut rng, beat_phase.fract() * spb, 0.05) as f64 * 0.5;
            s += hat(&mut rng, ((beat_phase + 0.5) % 1.0) * spb, 0.03) as f64 * 0.3;
        } else {
            let kt = beat_phase * spb;
            s += hat(&mut rng, kt, 0.05) as f64 * 0.4;
        }
        // bass on offbeat 8ths (classic DJ-friendly groove)
        let chord_pc = prog[(bar_idx.max(0) % 4) as usize];
        if !intro && o.heavy_bass {
            let bt = ((beat_phase + 0.5) % 1.0) * spb;
            if bt < 0.22 {
                let env = (-bt * 10.0).exp();
                s += (2.0 * std::f64::consts::PI * pc_freq(chord_pc, 2) * t).sin() * env * 0.7;
            }
        }
        // chord pad after intro
        if t > bar * 1.0 {
            let attack = ((t - bar) / 1.5).clamp(0.0, 1.0);
            s += chord_wave(t, &triad(chord_pc, o.minor), 4, 0.35 * attack);
        }
        // vocal-like line: narrow-band vibrato tone in mid range during "verse"
        if o.vocals && t > bar * 4.0 && t < o.duration_s - bar * 2.0 {
            let mel_pc = triad(chord_pc, o.minor)[(beat_idx.abs() % 3) as usize];
            let fv = pc_freq(mel_pc, 5);
            let vib = 1.0 + 0.006 * (2.0 * std::f64::consts::PI * 5.5 * t).sin();
            let env = (-(beat_phase * 3.0)).exp().max(0.2);
            s += (2.0 * std::f64::consts::PI * fv * vib * t).sin() * env * 0.22;
            // formant-ish second harmonic
            s += (2.0 * std::f64::consts::PI * fv * 2.01 * t).sin() * env * 0.08;
        }
        // global fades
        let fi = (t / 1.0).min(1.0); // 1 s fade-in
        let fo = ((o.duration_s - t) / 2.0).clamp(0.0, 1.0);
        let fade = if t > outro_start { fi * fo * fo } else { fi };
        buf[i] = (s * o.gain * fade).clamp(-1.0, 1.0) as f32;
    }
    AudioBuffer::new(fs, 1, buf)
}

/// Convenience: beat signal with given BPM & key name for docs/tests.
pub fn beat_signal(bpm: f64, key_name: &str, duration_s: f64, sample_rate: f64) -> AudioBuffer {
    let opts = SynthOptions {
        bpm,
        key_pc: name_to_pc(key_name),
        duration_s,
        sample_rate,
        ..Default::default()
    };
    synth_track(&opts)
}

/// Pure silence buffer.
pub fn silence(duration_s: f64, sample_rate: f64) -> AudioBuffer {
    AudioBuffer::new(sample_rate, 1, vec![0.0; (duration_s * sample_rate) as usize])
}

/// Ambient pad without any percussion (no clear beat): slow chords, noise.
pub fn ambient_track(duration_s: f64, sample_rate: f64, key_pc: usize) -> AudioBuffer {
    let mut rng = Rng::new(7);
    let n = (duration_s * sample_rate) as usize;
    let mut buf = vec![0.0f32; n];
    for i in 0..n {
        let t = i as f64 / sample_rate;
        let ch = (i / sample_rate as usize) % 3;
        let pc = [key_pc, (key_pc + 5) % 12, (key_pc + 3) % 12][ch];
        let mut s = chord_wave(t, &triad(pc, true), 3, 0.4) + chord_wave(t, &triad(pc, true), 4, 0.2);
        s += rng.f() * 0.02; // airy noise
        let fade = (t / 4.0).min(1.0) * ((duration_s - t) / 4.0).clamp(0.0, 1.0);
        buf[i] = (s * fade * 0.5) as f32;
    }
    AudioBuffer::new(sample_rate, 1, buf)
}

/// Spoken-word-ish: bursts of low-formant noise-banded energy with irregular
/// rhythm (used by edge-case tests).
pub fn spoken_word(duration_s: f64, sample_rate: f64) -> AudioBuffer {
    let mut rng = Rng::new(11);
    let n = (duration_s * sample_rate) as usize;
    let mut buf = vec![0.0f32; n];
    let mut active_until = 0.0;
    for i in 0..n {
        let t = i as f64 / sample_rate;
        if t > active_until {
            if rng.f() > 0.4 {
                active_until = t + 0.2 + rng.f().abs() * 1.2;
            } else {
                active_until = t + 0.4;
            }
        }
        let on = if t < active_until { 1.0 } else { 0.0 };
        let f = 150.0 + 60.0 * (2.0 * std::f64::consts::PI * 0.7 * t).sin();
        let s = (2.0 * std::f64::consts::PI * f * t).sin() * 0.3 + rng.f() * 0.05;
        buf[i] = (s * on) as f32;
    }
    AudioBuffer::new(sample_rate, 1, buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn synth_has_energy_and_beats() {
        let b = beat_signal(120.0, "A", 10.0, 22050.0);
        assert_eq!(b.samples.len(), 220500);
        let peak = b.samples.iter().fold(0.0f32, |m, x| m.max(x.abs()));
        assert!(peak > 0.2);
    }

    #[test]
    fn deterministic_same_seed() {
        let a = synth_track(&SynthOptions::default());
        let b = synth_track(&SynthOptions::default());
        assert_eq!(a.samples[..1000], b.samples[..1000]);
    }
}
