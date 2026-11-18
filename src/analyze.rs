//! Quality assessment, decision verdict and probability calibration.

use crate::embed;
use crate::scrfd::Face;
use image::RgbImage;
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Serialize, Deserialize)]
pub struct Calibration {
    pub slope: f32,
    pub mid: f32,
    pub threshold: f32,
}

impl Default for Calibration {
    fn default() -> Self { Self { slope: 14.0, mid: 0.40, threshold: 0.40 } }
}

pub fn probability(cos: f32, c: &Calibration) -> f32 {
    1.0 / (1.0 + (-c.slope * (cos - c.mid)).exp())
}

#[derive(Clone, Copy, PartialEq)]
pub enum Level { StrongSame, Same, Unsure, Diff, StrongDiff }

pub struct Verdict {
    pub cos: f32, pub l2: f32, pub prob: f32,
    pub level: Level, pub label: &'static str,
}

pub fn verdict(a: &[f32], b: &[f32], c: &Calibration) -> Verdict {
    let cos = embed::cosine(a, b);
    let l2 = (2.0 - 2.0 * cos).max(0.0).sqrt();
    let t = c.threshold;

    let (level, label) = if cos >= t + 0.10 {
        (Level::StrongSame, "VERY LIKELY THE SAME PERSON")
    } else if cos >= t {
        (Level::Same, "LIKELY THE SAME PERSON")
    } else if cos >= t - 0.08 {
        (Level::Unsure, "INCONCLUSIVE - RETAKE THE PHOTOS")
    } else if cos >= t - 0.20 {
        (Level::Diff, "LIKELY DIFFERENT PEOPLE")
    } else {
        (Level::StrongDiff, "VERY LIKELY DIFFERENT PEOPLE")
    };

    Verdict { cos, l2, prob: probability(cos, c), level, label }
}

#[derive(Clone, Default)]
pub struct Quality {
    pub size: f32, pub det: f32, pub blur: f32, pub bright: f32, pub contrast: f32,
    pub yaw: f32, pub roll: f32, pub pitch: f32,
    pub warnings: Vec<String>,
}

pub fn assess(f: &Face, aligned: &RgbImage) -> Quality {
    let (w, h) = (aligned.width() as usize, aligned.height() as usize);
    let g: Vec<f32> = aligned.pixels().map(|p| 0.299 * p[0] as f32 + 0.587 * p[1] as f32 + 0.114 * p[2] as f32).collect();
    let n = g.len().max(1) as f32;
    let bright = g.iter().sum::<f32>() / n;
    let contrast = (g.iter().map(|v| (v - bright).powi(2)).sum::<f32>() / n).sqrt();

    let mut lap = Vec::new();
    for y in 1..h.saturating_sub(1) {
        for x in 1..w.saturating_sub(1) {
            let c = g[y * w + x];
            lap.push(g[(y - 1) * w + x] + g[(y + 1) * w + x] + g[y * w + x - 1] + g[y * w + x + 1] - 4.0 * c);
        }
    }
    let ln = lap.len().max(1) as f32;
    let lm = lap.iter().sum::<f32>() / ln;
    let blur = lap.iter().map(|v| (v - lm).powi(2)).sum::<f32>() / ln;

    let k = &f.kps;
    let eye_mid = [(k[0][0] + k[1][0]) / 2.0, (k[0][1] + k[1][1]) / 2.0];
    let mouth_mid = [(k[3][0] + k[4][0]) / 2.0, (k[3][1] + k[4][1]) / 2.0];
    let eye_d = ((k[1][0] - k[0][0]).powi(2) + (k[1][1] - k[0][1]).powi(2)).sqrt().max(1e-3);
    let yaw = (k[2][0] - eye_mid[0]) / eye_d;
    let roll = (k[1][1] - k[0][1]).atan2(k[1][0] - k[0][0]).to_degrees();
    let denom = (k[2][1] - eye_mid[1]).abs().max(1e-3);
    let pitch = (mouth_mid[1] - k[2][1]) / denom;

    let size = f.short_side();
    let mut warnings = Vec::new();
    if size < 60.0 { warnings.push(format!("face is tiny ({size:.0}px) - result not reliable")); }
    else if size < 90.0 { warnings.push(format!("face is small ({size:.0}px) - aim for >= 110px")); }
    if f.score < 0.75 { warnings.push("detector confidence is low".into()); }
    if blur < 25.0 { warnings.push("image is blurred / out of focus".into()); }
    else if blur < 60.0 { warnings.push("image is soft".into()); }
    if bright < 55.0 { warnings.push("too dark".into()); }
    if bright > 205.0 { warnings.push("too bright / overexposed".into()); }
    if contrast < 25.0 { warnings.push("low contrast (backlight or haze)".into()); }
    if yaw.abs() > 0.28 { warnings.push("head is turned substantially".into()); }
    if roll.abs() > 22.0 { warnings.push("head is tilted (roll)".into()); }
    if !(0.65..=1.55).contains(&pitch) { warnings.push("head is pitched up or down".into()); }

    Quality { size, det: f.score, blur, bright, contrast, yaw, roll, pitch, warnings }
}