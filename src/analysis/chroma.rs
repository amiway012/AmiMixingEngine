//! Stage 3: tonal analysis — HPCP chroma, Krumhansl-Schmuckler key detection,
//! modulation scanning, bass roots and melodic density.

use crate::analysis::Stft;
use crate::model::*;

/// Krumhansl-Kessler profiles (1982).
const KS_MAJOR: [f64; 12] = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88];
const KS_MINOR: [f64; 12] = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17];

/// Pearson correlation between two vectors.
pub fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len() as f64;
    let ma = a.iter().sum::<f64>() / n;
    let mb = b.iter().sum::<f64>() / n;
    let mut cab = 0.0;
    let mut caa = 0.0;
    let mut cbb = 0.0;
    for i in 0..a.len() {
        let x = a[i] - ma;
        let y = b[i] - mb;
        cab += x * y;
        caa += x * x;
        cbb += y * y;
    }
    if caa <= 1e-12 || cbb <= 1e-12 {
        0.0
    } else {
        cab / (caa.sqrt() * cbb.sqrt())
    }
}

/// Fold one power spectrum into 12 chroma bins (HPCP-style): spectral peaks
/// are parabolically interpolated, folded mod 12 semitones, harmonically weighted.
pub fn chroma_from_spectrum(ps: &[f64], sample_rate: f64, n_fft: usize) -> [f64; 12] {
    let mut c = [0.0f64; 12];
    let max = ps.iter().cloned().fold(0.0f64, f64::max);
    if max <= 1e-20 {
        return c;
    }
    let thresh = max * 1e-4;
    for k in 2..ps.len().saturating_sub(1) {
        let v = ps[k];
        if v < thresh || v < ps[k - 1] || v > ps[k + 1] {
            continue; // not a local maximum
        }
        let f = k as f64 * sample_rate / n_fft as f64;
        if f < 55.0 || f > 8000.0 {
            continue;
        }
        let denom = ps[k - 1] - 2.0 * v + ps[k + 1];
        let delta = if denom.abs() > 1e-12 { 0.5 * (ps[k - 1] - ps[k + 1]) / denom } else { 0.0 };
        let fk = (k as f64 + delta.clamp(-0.5, 0.5)) * sample_rate / n_fft as f64;
        let midi = 69.0 + 12.0 * (fk / 440.0).log2();
        let pc = (midi.round() as i64).rem_euclid(12) as usize;
        let w = v.sqrt() / (1.0 + f / 1000.0);
        c[pc] += w;
    }
    c
}

/// KS key classification on an averaged chroma. Returns (pc, scale, margin).
pub fn detect_key(chroma: &[f64; 12]) -> (usize, Scale, f64) {
    let mut best = (0usize, Scale::Major, f64::NEG_INFINITY);
    let mut second = f64::NEG_INFINITY;
    for pc in 0..12 {
        for scale in [Scale::Major, Scale::Minor] {
            let profile = match scale {
                Scale::Major => &KS_MAJOR,
                Scale::Minor => &KS_MINOR,
            };
            let mut rotated = vec![0.0f64; 12];
            for i in 0..12 {
                rotated[(i + pc) % 12] = profile[i];
            }
            let r = pearson(chroma, &rotated);
            if r > best.2 {
                second = best.2;
                best.2 = r;
                best.0 = pc;
                best.1 = scale;
            } else if r > second {
                second = r;
            }
        }
    }
    let margin = (best.2 - second).max(0.0);
    (best.0, best.1, margin)
}

/// Bass root estimate from the low band of a power spectrum.
fn bass_root_ps(ps: &[f64], sample_rate: f64, n_fft: usize) -> Option<usize> {
    let lo_bin = (40.0 * n_fft as f64 / sample_rate) as usize;
    let hi_bin = ((160.0 * n_fft as f64 / sample_rate) as usize).min(ps.len().saturating_sub(1));
    if hi_bin <= lo_bin {
        return None;
    }
    let (mut bi, mut bv) = (lo_bin, 0.0f64);
    for k in lo_bin..hi_bin {
        if ps[k] > bv {
            bv = ps[k];
            bi = k;
        }
    }
    if bv <= 1e-12 {
        return None;
    }
    let f = bi as f64 * sample_rate / n_fft as f64;
    let midi = 69.0 + 12.0 * (f / 440.0).log2();
    Some((midi.round() as i64).rem_euclid(12) as usize)
}

/// Melodic density proxy: normalized entropy complement of chroma profile.
fn melodic_density(c: &[f64; 12]) -> f64 {
    let s: f64 = c.iter().sum();
    if s <= 1e-12 {
        return 0.0;
    }
    let h: f64 = c
        .iter()
        .map(|x| {
            let p = x / s;
            if p > 1e-12 {
                -p * p.ln()
            } else {
                0.0
            }
        })
        .sum();
    let hmax = 12.0f64.ln();
    (1.0 - h / hmax).clamp(0.0, 1.0)
}

/// Full stage-3 pipeline.
pub fn analyze_harmony(stft: &mut Stft, mono: &[f32], _sr: f64, duration: f64, rhythm: &TrackRhythm) -> TrackHarmony {
    let sr = stft.sample_rate;
    let nfft = stft.fft.size();
    let nf = stft.n_frames(mono.len());
    if nf == 0 {
        return TrackHarmony::default();
    }
    // one pass: store per-frame chroma + times
    let mut times = Vec::with_capacity(nf);
    let mut frames: Vec<[f64; 12]> = Vec::with_capacity(nf);
    stft.each_frame(mono, |f, ps| {
        times.push(stft.frame_time(f));
        frames.push(chroma_from_spectrum(ps, sr, nfft));
    });

    let mut global = [0.0f64; 12];
    for fr in &frames {
        for i in 0..12 {
            global[i] += fr[i];
        }
    }
    let gmax = global.iter().cloned().fold(0.0f64, f64::max).max(1e-12);
    for i in 0..12 {
        global[i] /= gmax;
    }
    let (pc, scale, margin) = detect_key(&global);
    let conf = (margin * 3.0 + 0.15).clamp(0.05, 0.98);

    // ---- modulation scan over ~8-bar windows ------------------------------
    let mut modulations = Vec::new();
    let seg_len = if rhythm.bpm > 0.0 {
        (60.0 / rhythm.bpm) * 4.0 * 8.0
    } else {
        15.0
    };
    let seg_len = seg_len.min((duration / 3.0).max(6.0));
    let mut prev_key: Option<(usize, Scale)> = None;
    let mut densities = Vec::new();
    let mut a = 0.0f64;
    while a < duration {
        let b = (a + seg_len).min(duration);
        let mut acc = [0.0f64; 12];
        let mut cnt = 0usize;
        for (ti, fr) in times.iter().zip(frames.iter()) {
            if *ti >= a && *ti < b {
                for i in 0..12 {
                    acc[i] += fr[i];
                }
                cnt += 1;
            }
        }
        if cnt > 4 {
            let amax = acc.iter().cloned().fold(0.0f64, f64::max).max(1e-12);
            for i in 0..12 {
                acc[i] /= amax;
            }
            let (spc, sscale, smargin) = detect_key(&acc);
            if let Some((ppc, pscale)) = prev_key {
                // require both change AND decent margin to avoid noise flips
                if (spc != ppc || sscale != pscale) && smargin > 0.08 {
                    modulations.push(Modulation {
                        time: a,
                        from_key: format!("{}{}", pc_to_name(ppc), if pscale == Scale::Minor { "m" } else { "" }),
                        to_key: format!("{}{}", pc_to_name(spc), if sscale == Scale::Minor { "m" } else { "" }),
                    });
                }
            }
            prev_key = Some((spc, sscale));
            densities.push(melodic_density(&acc));
        }
        a = b;
        if b >= duration {
            break;
        }
    }

    // ---- bass notes per quarter -------------------------------------------
    let mut bass_notes = Vec::new();
    let mut ps_buf = vec![0.0f64; stft.bins()];
    for q in 0..4 {
        let lo = nf * q / 4;
        let hi = ((nf * (q + 1) / 4).max(lo + 1)).min(nf);
        let mut votes = [0usize; 12];
        for f in lo..hi {
            stft.frame_power_into(mono, f, &mut ps_buf);
            if let Some(bpc) = bass_root_ps(&ps_buf, sr, nfft) {
                votes[bpc] += 1;
            }
        }
        let pick = votes.iter().enumerate().max_by_key(|(_, v)| *v).filter(|(_, v)| **v > 0).map(|(i, _)| i);
        bass_notes.push(pick.unwrap_or(pc) as f64);
    }

    TrackHarmony {
        key: pc_to_name(pc).into(),
        scale,
        camelot: camelot_for(pc, scale),
        confidence: conf,
        chroma: global,
        modulations,
        bass_notes,
        melodic_density: densities,
    }
}

/// Harmonic compatibility 0..1 between two harmonies (Camelot wheel rules):
/// identical → 1.0; ±1 neighbor → 0.9; relative major/minor → 0.75; distant → low.
pub fn harmonic_compatibility(a: &TrackHarmony, b: &TrackHarmony) -> f64 {
    let d = camelot_distance(&a.camelot, &b.camelot);
    let la = parse_camelot(&a.camelot).map(|x| x.1);
    let lb = parse_camelot(&b.camelot).map(|x| x.1);
    let same_letter = la == lb;
    if d == 0 && !same_letter {
        return 0.75; // relative major/minor
    }
    let base = match d {
        0 => 1.0,
        1 => 0.9,
        2 => 0.6,
        3 => 0.45,
        4 => 0.3,
        5 => 0.2,
        _ => 0.1,
    };
    // chroma-correlation sanity check blends template evidence with wheel rules
    let cor = pearson(&a.chroma, &b.chroma);
    (base * 0.75 + cor.max(0.0) * 0.25).clamp(0.05, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::synth;

    #[test]
    fn detects_a_minor() {
        let b = synth::beat_signal(124.0, "A", 24.0, 22050.0);
        let mut stft = Stft::new(22050.0, 2048, 512);
        let mono = b.mono();
        let r = TrackRhythm::default();
        let h = analyze_harmony(&mut stft, &mono, 22050.0, b.duration(), &r);
        assert_eq!(h.key, "A", "got {}", h.key);
        assert_eq!(h.scale, Scale::Minor);
        assert_eq!(h.camelot, "8A");
    }

    #[test]
    fn camelot_rules() {
        let mk = |name: &str, minor: bool| -> TrackHarmony {
            let pc = synth::name_to_pc(name);
            let scale = if minor { Scale::Minor } else { Scale::Major };
            let tri = synth::triad(pc, minor);
            let mut c = [0.2f64; 12];
            for t in tri {
                c[t] = 1.0;
            }
            TrackHarmony {
                key: name.into(),
                scale,
                camelot: camelot_for(pc, scale),
                chroma: c,
                ..Default::default()
            }
        };
        assert!(harmonic_compatibility(&mk("A", true), &mk("A", true)) > 0.9);
        assert!(harmonic_compatibility(&mk("A", true), &mk("E", true))
            > harmonic_compatibility(&mk("A", true), &mk("C#", true)));
        assert!(harmonic_compatibility(&mk("A", true), &mk("C", false)) > 0.6); // relative
    }
}
