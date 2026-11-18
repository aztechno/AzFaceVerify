//! SCRFD face detector: bounding boxes plus 5 facial landmarks.

use crate::engine::{build_session, input_name, run_f32, Provider};
use anyhow::{anyhow, Result};
use image::{imageops::FilterType, RgbImage};
use ort::session::Session;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct Face {
    pub bbox: [f32; 4],
    pub score: f32,
    pub kps: [[f32; 2]; 5],
}

impl Face {
    pub fn w(&self) -> f32 { self.bbox[2] - self.bbox[0] }
    pub fn h(&self) -> f32 { self.bbox[3] - self.bbox[1] }
    pub fn area(&self) -> f32 { self.w().max(0.0) * self.h().max(0.0) }
    pub fn short_side(&self) -> f32 { self.w().min(self.h()) }
}

pub struct Detector {
    session: Session,
    in_name: String,
    size: usize,
    strides: [usize; 3],
    anchors: usize,
}

impl Detector {
    pub fn new(model: &Path, p: Provider, threads: usize, size: usize) -> Result<Self> {
        let session = build_session(model, p, threads)?;
        let in_name = input_name(&session);
        Ok(Self {
            session,
            in_name,
            size,
            strides: [8, 16, 32],
            anchors: 2,
        })
    }

    /// Note: takes &mut self because ort 2.x requires &mut Session for run()
    pub fn detect(&mut self, img: &RgbImage, thr: f32, nms_thr: f32) -> Result<Vec<Face>> {
        let (iw, ih) = (img.width() as f32, img.height() as f32);
        let m = self.size;

        let ratio = (m as f32 / iw).min(m as f32 / ih);
        let nw = ((iw * ratio).round() as usize).clamp(1, m);
        let nh = ((ih * ratio).round() as usize).clamp(1, m);
        let small = image::imageops::resize(img, nw as u32, nh as u32, FilterType::Triangle);

        let plane = m * m;
        let pad = -127.5 / 128.0;
        let mut data = vec![pad; 3 * plane];
        for y in 0..nh {
            for x in 0..nw {
                let px = small.get_pixel(x as u32, y as u32).0;
                let i = y * m + x;
                data[i] = (px[0] as f32 - 127.5) / 128.0;
                data[plane + i] = (px[1] as f32 - 127.5) / 128.0;
                data[2 * plane + i] = (px[2] as f32 - 127.5) / 128.0;
            }
        }

        let out = run_f32(
            &mut self.session,
            &self.in_name,
            &[1, 3, m as i64, m as i64],
            data,
            9,
        )?;
        let map = output_layout(&out)?;

        let mut candidates: Vec<Face> = Vec::new();
        for (si, &stride) in self.strides.iter().enumerate() {
            let (sc, bb, kp) = (map[si][0], map[si][1], map[si][2]);
            let (fh, fw, na) = (m / stride, m / stride, self.anchors);
            let n = fh * fw * na;
            if out[sc].len() != n || out[bb].len() != 4 * n || out[kp].len() != 10 * n {
                return Err(anyhow!("unexpected SCRFD output shapes at stride {stride}"));
            }

            let (s, b, k) = (&out[sc], &out[bb], &out[kp]);
            let sf = stride as f32;

            for y in 0..fh {
                for x in 0..fw {
                    let base = (y * fw + x) * na;
                    for a in 0..na {
                        let idx = base + a;
                        let score = s[idx];
                        if score < thr {
                            continue;
                        }
                        let (cx, cy) = ((x * stride) as f32, (y * stride) as f32);
                        let (bi, ki) = (idx * 4, idx * 10);

                        let mut kps = [[0.0f32; 2]; 5];
                        for j in 0..5 {
                            kps[j] = [cx + k[ki + 2 * j] * sf, cy + k[ki + 2 * j + 1] * sf];
                        }

                        candidates.push(Face {
                            bbox: [
                                cx - b[bi] * sf,
                                cy - b[bi + 1] * sf,
                                cx + b[bi + 2] * sf,
                                cy + b[bi + 3] * sf,
                            ],
                            score,
                            kps,
                        });
                    }
                }
            }
        }

        let mut keep = nms(candidates, nms_thr);
        let inv = 1.0 / ratio;
        for f in keep.iter_mut() {
            f.bbox[0] = (f.bbox[0] * inv).clamp(0.0, iw);
            f.bbox[1] = (f.bbox[1] * inv).clamp(0.0, ih);
            f.bbox[2] = (f.bbox[2] * inv).clamp(0.0, iw);
            f.bbox[3] = (f.bbox[3] * inv).clamp(0.0, ih);
            for p in f.kps.iter_mut() {
                p[0] *= inv;
                p[1] *= inv;
            }
        }

        keep.sort_by(|a, b| b.area().partial_cmp(&a.area()).unwrap_or(std::cmp::Ordering::Equal));
        Ok(keep)
    }
}

fn output_layout(out: &[Vec<f32>]) -> Result<[[usize; 3]; 3]> {
    if out.len() != 9 {
        return Err(anyhow!("detector must expose 9 outputs (SCRFD), found {}", out.len()));
    }
    for c in [
        [[0, 3, 6], [1, 4, 7], [2, 5, 8]],
        [[0, 1, 2], [3, 4, 5], [6, 7, 8]],
    ] {
        let ratios_ok = (0..3).all(|i| {
            let n = out[c[i][0]].len();
            n > 0 && out[c[i][1]].len() == 4 * n && out[c[i][2]].len() == 10 * n
        });
        let order_ok = out[c[0][0]].len() > out[c[1][0]].len()
            && out[c[1][0]].len() > out[c[2][0]].len();
        if ratios_ok && order_ok {
            return Ok(c);
        }
    }
    Err(anyhow!("unable to recognise SCRFD output ordering"))
}

fn iou(a: &[f32; 4], b: &[f32; 4]) -> f32 {
    let (x1, y1) = (a[0].max(b[0]), a[1].max(b[1]));
    let (x2, y2) = (a[2].min(b[2]), a[3].min(b[3]));
    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let union = (a[2] - a[0]).max(0.0) * (a[3] - a[1]).max(0.0)
        + (b[2] - b[0]).max(0.0) * (b[3] - b[1]).max(0.0)
        - inter;
    if union <= 0.0 { 0.0 } else { inter / union }
}

fn nms(mut faces: Vec<Face>, thr: f32) -> Vec<Face> {
    faces.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    let mut keep: Vec<Face> = Vec::new();
    for c in faces {
        if keep.iter().all(|k| iou(&k.bbox, &c.bbox) < thr) {
            keep.push(c);
        }
    }
    keep
}