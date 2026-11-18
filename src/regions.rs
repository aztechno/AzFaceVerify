//! Part-based ("which facial area matters") analysis for AzFaceVerify.
//! Built by Azis Ichwan.
//!
//! ============================ HONESTY NOTE ============================
//! ArcFace outputs ONE global 512-d embedding. It does NOT emit per-organ
//! similarity. Everything in this module is DERIVED, not read from the model:
//!
//!  1. Occlusion attribution (most defensible).
//!     Hide the same region in BOTH aligned faces, re-run the model, and
//!     measure how the global cosine changes:
//!         delta = cos_baseline - cos_with_region_hidden
//!     delta > 0  -> that region was pushing the score toward "same person"
//!     delta < 0  -> that region was pushing the score toward "different"
//!     This is standard occlusion-based interpretability. Caveat: masking also
//!     pushes the input out of the training distribution, and regions overlap,
//!     so delta is an approximation, not an exact decomposition.
//!
//!  2. Region-only similarity (weaker).
//!     Keep only one region visible in both faces and compare. The model was
//!     trained on full faces, so these embeddings are out-of-distribution.
//!     Use for RANKING regions only. NEVER compare these numbers against the
//!     full-face decision threshold.
//!
//!  3. Landmark geometry (weak for identity).
//!     Ratios from 5 landmarks, normalised by inter-ocular distance. Heavily
//!     affected by pose and expression. Useful to explain alignment/expression
//!     differences, NOT as identity evidence on its own.
//!
//!  4. Pixel difference heatmap (visual only).
//!     Dominated by lighting, pose and expression. NOT identity evidence.
//! =====================================================================

use crate::scrfd::Face;
use image::{Rgb, RgbImage};

/// ArcFace aligned crop size.
pub const CROP: usize = 112;

/// Neutral fill. 128 maps to ~0.0 after (x - 127.5) / 127.5 normalisation.
const FILL: Rgb<u8> = Rgb([128, 128, 128]);

#[derive(Clone, Copy)]
pub struct Region {
    pub name: &'static str,
    /// x0, y0, x1, y1 inside the 112x112 aligned crop.
    /// Derived from the ArcFace template: eyes y~51.6, nose y~71.7, mouth y~92.3.
    pub rect: [usize; 4],
    pub note: &'static str,
}

pub const REGIONS: &[Region] = &[
    Region {
        name: "Eyes + eyebrows",
        rect: [16, 34, 96, 64],
        note: "Usually the strongest identity-bearing area for ArcFace. Sunglasses or a fringe that covers this area hurts accuracy the most.",
    },
    Region {
        name: "Nose",
        rect: [40, 60, 72, 84],
        note: "Very stable across expression and facial hair, but sensitive to head pose (yaw).",
    },
    Region {
        name: "Mouth",
        rect: [30, 80, 82, 104],
        note: "Expression, smiling, moustache and lipstick live here. A negative delta often means expression difference, not a different person.",
    },
    Region {
        name: "Chin + jawline",
        rect: [8, 96, 104, 112],
        note: "Beard growth, weight change and head pitch change this area a lot.",
    },
    Region {
        name: "Forehead + hairline",
        rect: [8, 4, 104, 34],
        note: "Hats, caps, hijab and fringe cover this. Usually a weak identity contributor.",
    },
];

/// Hide the region (fill it with neutral grey), keep everything else.
pub fn occlude(src: &RgbImage, rect: &[usize; 4]) -> RgbImage {
    let mut out = src.clone();
    let (w, h) = (out.width() as usize, out.height() as usize);
    for y in rect[1]..rect[3] {
        for x in rect[0]..rect[2] {
            if x < w && y < h {
                out.put_pixel(x as u32, y as u32, FILL);
            }
        }
    }
    out
}

/// Keep only the region, neutralise everything else.
pub fn isolate(src: &RgbImage, rect: &[usize; 4]) -> RgbImage {
    let mut out = RgbImage::from_pixel(src.width(), src.height(), FILL);
    let (w, h) = (src.width() as usize, src.height() as usize);
    for y in rect[1]..rect[3] {
        for x in rect[0]..rect[2] {
            if x < w && y < h {
                out.put_pixel(x as u32, y as u32, *src.get_pixel(x as u32, y as u32));
            }
        }
    }
    out
}

/// Absolute luminance difference, normalised and colour-mapped.
/// VISUAL ONLY - dominated by lighting, pose and expression.
pub fn diff_heatmap(a: &RgbImage, b: &RgbImage) -> RgbImage {
    let w = a.width().min(b.width());
    let h = a.height().min(b.height());
    if w == 0 || h == 0 {
        return RgbImage::new(CROP as u32, CROP as u32);
    }

    let mut d = vec![0f32; (w * h) as usize];
    let mut max = 1e-6f32;
    for y in 0..h {
        for x in 0..w {
            let pa = a.get_pixel(x, y).0;
            let pb = b.get_pixel(x, y).0;
            let la = 0.299 * pa[0] as f32 + 0.587 * pa[1] as f32 + 0.114 * pa[2] as f32;
            let lb = 0.299 * pb[0] as f32 + 0.587 * pb[1] as f32 + 0.114 * pb[2] as f32;
            let v = (la - lb).abs();
            d[(y * w + x) as usize] = v;
            if v > max {
                max = v;
            }
        }
    }

    let mut out = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let t = (d[(y * w + x) as usize] / max).clamp(0.0, 1.0);
            out.put_pixel(x, y, ramp(t));
        }
    }
    out
}

fn ramp(t: f32) -> Rgb<u8> {
    let stops: [(f32, [f32; 3]); 5] = [
        (0.00, [20.0, 30.0, 90.0]),
        (0.25, [20.0, 160.0, 200.0]),
        (0.50, [40.0, 200.0, 90.0]),
        (0.75, [240.0, 210.0, 60.0]),
        (1.00, [230.0, 60.0, 50.0]),
    ];
    for i in 0..stops.len() - 1 {
        let (t0, c0) = stops[i];
        let (t1, c1) = stops[i + 1];
        if t >= t0 && t <= t1 {
            let f = if t1 > t0 { (t - t0) / (t1 - t0) } else { 0.0 };
            let mut c = [0u8; 3];
            for k in 0..3 {
                c[k] = (c0[k] + (c1[k] - c0[k]) * f).round().clamp(0.0, 255.0) as u8;
            }
            return Rgb(c);
        }
    }
    Rgb([230, 60, 50])
}

pub const GEO_LABELS: [(&str, &str); 8] = [
    ("Inter-ocular distance", "raw pixels - depends on image scale, NOT identity"),
    ("Nose height / eye distance", "how far the nose tip sits below the eye line"),
    ("Mouth width / eye distance", "mouth corner span"),
    ("Nose to mouth / eye distance", "lower-face vertical proportion"),
    ("Eye line to mouth / eye distance", "overall face height proportion"),
    ("Nose horizontal offset / eye distance", "yaw / asymmetry proxy"),
    ("Mouth horizontal offset / eye distance", "yaw / expression proxy"),
    ("Eye line roll", "degrees - head tilt"),
];

/// 8 coarse geometric descriptors from the 5 landmarks.
pub fn geometry(f: &Face) -> [f32; 8] {
    let k = &f.kps;
    let (le, re, no, lm, rm) = (k[0], k[1], k[2], k[3], k[4]);
    let em = [(le[0] + re[0]) / 2.0, (le[1] + re[1]) / 2.0];
    let mm = [(lm[0] + rm[0]) / 2.0, (lm[1] + rm[1]) / 2.0];
    let ed = ((re[0] - le[0]).powi(2) + (re[1] - le[1]).powi(2))
        .sqrt()
        .max(1e-3);
    let mw = ((rm[0] - lm[0]).powi(2) + (rm[1] - lm[1]).powi(2)).sqrt();

    [
        ed,
        (no[1] - em[1]) / ed,
        mw / ed,
        (mm[1] - no[1]) / ed,
        (mm[1] - em[1]) / ed,
        (no[0] - em[0]) / ed,
        (mm[0] - em[0]) / ed,
        (re[1] - le[1]).atan2(re[0] - le[0]).to_degrees(),
    ]
}

#[derive(Clone)]
pub struct RegionScore {
    pub name: &'static str,
    pub note: &'static str,
    /// cosine when this region is hidden in BOTH faces
    pub cos_occluded: f32,
    /// cos_baseline - cos_occluded
    pub delta: f32,
    /// cosine when ONLY this region is visible in both faces (out-of-distribution)
    pub cos_region_only: f32,
}

#[derive(Clone)]
pub struct DeepReport {
    /// headline score of the pair (may include flip-TTA)
    pub cos_headline: f32,
    /// no-TTA baseline used for all deltas (apples-to-apples comparison)
    pub cos_base: f32,
    pub regions: Vec<RegionScore>,
    pub geo_a: [f32; 8],
    pub geo_b: [f32; 8],
    pub aligned_a: RgbImage,
    pub aligned_b: RgbImage,
    pub heatmap: RgbImage,
    pub ms: f32,
    pub forwards: usize,
}

impl DeepReport {
    pub fn strongest_for(&self) -> Option<&RegionScore> {
        self.regions
            .iter()
            .filter(|r| r.delta > 0.0)
            .max_by(|a, b| a.delta.partial_cmp(&b.delta).unwrap_or(std::cmp::Ordering::Equal))
    }

    pub fn strongest_against(&self) -> Option<&RegionScore> {
        self.regions
            .iter()
            .filter(|r| r.delta < 0.0)
            .min_by(|a, b| a.delta.partial_cmp(&b.delta).unwrap_or(std::cmp::Ordering::Equal))
    }
}

pub fn interpret(delta: f32) -> &'static str {
    if delta >= 0.10 {
        "strong evidence FOR same person"
    } else if delta >= 0.03 {
        "supports same person"
    } else if delta > -0.03 {
        "little influence"
    } else if delta > -0.10 {
        "pushes toward different"
    } else {
        "strong push toward different"
    }
}