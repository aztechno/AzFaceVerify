//! Face alignment (5-point similarity transform) plus ArcFace embedding.

use crate::engine::{build_session, input_name, run_f32, Provider};
use crate::scrfd::Face;
use anyhow::{anyhow, Result};
use image::{Rgb, RgbImage};
use ort::session::Session;
use std::path::Path;

pub const OUT: usize = 112;

pub const TEMPLATE: [[f32; 2]; 5] = [
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
];

pub fn similarity(src: &[[f32; 2]; 5], dst: &[[f32; 2]; 5]) -> [f64; 6] {
    let n = 5.0f64;
    let (mut ms, mut md) = ([0.0f64; 2], [0.0f64; 2]);
    for i in 0..5 {
        for k in 0..2 {
            ms[k] += src[i][k] as f64 / n;
            md[k] += dst[i][k] as f64 / n;
        }
    }

    let (mut a, mut b, mut norm) = (0.0f64, 0.0f64, 0.0f64);
    for i in 0..5 {
        let p = [src[i][0] as f64 - ms[0], src[i][1] as f64 - ms[1]];
        let q = [dst[i][0] as f64 - md[0], dst[i][1] as f64 - md[1]];
        a += p[0] * q[0] + p[1] * q[1];
        b += p[0] * q[1] - p[1] * q[0];
        norm += p[0] * p[0] + p[1] * p[1];
    }

    let theta = b.atan2(a);
    let scale = if norm > 1e-9 { (a * a + b * b).sqrt() / norm } else { 1.0 };
    let (s, c) = (theta.sin() * scale, theta.cos() * scale);
    let tx = md[0] - (c * ms[0] - s * ms[1]);
    let ty = md[1] - (s * ms[0] + c * ms[1]);
    [c, -s, tx, s, c, ty]
}

pub fn warp(img: &RgbImage, m: &[f64; 6], w: u32, h: u32) -> RgbImage {
    let det = m[0] * m[4] - m[1] * m[3];
    let inv = if det.abs() < 1e-12 { [1.0, 0.0, 0.0, 1.0] } else {
        [m[4] / det, -m[1] / det, -m[3] / det, m[0] / det]
    };

    let (iw, ih) = (img.width() as i32, img.height() as i32);
    let mut out = RgbImage::new(w, h);
    for y in 0..h {
        for x in 0..w {
            let dx = x as f64 - m[2];
            let dy = y as f64 - m[5];
            out.put_pixel(x, y, bilinear(img, inv[0] * dx + inv[1] * dy, inv[2] * dx + inv[3] * dy, iw, ih));
        }
    }
    out
}

fn bilinear(img: &RgbImage, x: f64, y: f64, iw: i32, ih: i32) -> Rgb<u8> {
    if x < -1.0 || y < -1.0 || x > iw as f64 || y > ih as f64 { return Rgb([0, 0, 0]); }
    let (x0, y0) = (x.floor() as i32, y.floor() as i32);
    let (fx, fy) = ((x - x0 as f64) as f32, (y - y0 as f64) as f32);

    let get = |xx: i32, yy: i32| -> [f32; 3] {
        let p = img.get_pixel(xx.clamp(0, iw - 1) as u32, yy.clamp(0, ih - 1) as u32).0;
        [p[0] as f32, p[1] as f32, p[2] as f32]
    };

    let (p00, p01, p10, p11) = (get(x0, y0), get(x0 + 1, y0), get(x0, y0 + 1), get(x0 + 1, y0 + 1));
    let mut c = [0u8; 3];
    for k in 0..3 {
        let v = (p00[k] * (1.0 - fx) + p01[k] * fx) * (1.0 - fy) + (p10[k] * (1.0 - fx) + p11[k] * fx) * fy;
        c[k] = v.round().clamp(0.0, 255.0) as u8;
    }
    Rgb(c)
}

pub fn align(img: &RgbImage, f: &Face) -> RgbImage {
    warp(img, &similarity(&f.kps, &TEMPLATE), OUT as u32, OUT as u32)
}

pub struct Recognizer {
    session: Session,
    in_name: String,
    dim: usize,
}

impl Recognizer {
    pub fn new(model: &Path, p: Provider, threads: usize) -> Result<Self> {
        let session = build_session(model, p, threads)?;
        let in_name = input_name(&session);
        Ok(Self { session, in_name, dim: 512 })
    }

    /// Note: takes &mut self because ort 2.x requires &mut Session for run()
    fn forward(&mut self, crop: &RgbImage) -> Result<Vec<f32>> {
        if crop.width() != OUT as u32 || crop.height() != OUT as u32 {
            return Err(anyhow!("crop must be 112x112"));
        }
        let plane = OUT * OUT;
        let mut d = vec![0f32; 3 * plane];
        for y in 0..OUT {
            for x in 0..OUT {
                let p = crop.get_pixel(x as u32, y as u32).0;
                let i = y * OUT + x;
                d[i] = (p[0] as f32 - 127.5) / 127.5;
                d[plane + i] = (p[1] as f32 - 127.5) / 127.5;
                d[2 * plane + i] = (p[2] as f32 - 127.5) / 127.5;
            }
        }

        let mut o = run_f32(&mut self.session, &self.in_name, &[1, 3, OUT as i64, OUT as i64], d, 1)?;
        let e = o.pop().ok_or_else(|| anyhow!("empty embedding output"))?;
        if e.len() != self.dim {
            return Err(anyhow!("embedding dim is {} (expected 512)", e.len()));
        }
        Ok(e)
    }

    pub fn embed(&mut self, crop: &RgbImage, flip_tta: bool) -> Result<Vec<f32>> {
        let mut e = self.forward(crop)?;
        if flip_tta {
            let e2 = self.forward(&image::imageops::flip_horizontal(crop))?;
            for i in 0..e.len() { e[i] += e2[i]; }
        }
        normalize(&mut e);
        Ok(e)
    }
}

pub fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-12);
    for x in v.iter_mut() { *x /= n; }
}

pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>().clamp(-1.0, 1.0)
}