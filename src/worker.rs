//! Background inference worker. The GUI thread never blocks on the model.
//! Built by Azis Ichwan.

use crate::analyze::{self, Quality};
use crate::embed::{self, Recognizer};
use crate::engine::Provider;
use crate::regions::{self, DeepReport, RegionScore};
use crate::scrfd::{Detector, Face};
use anyhow::Result;
use image::RgbImage;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::time::Instant;

#[derive(Clone, Copy, PartialEq)]
pub struct Settings {
    pub provider: Provider,
    pub det_size: usize,
    pub det_thresh: f32,
    pub nms: f32,
    pub flip_tta: bool,
    pub threads: usize,
    pub max_faces: usize,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            provider: Provider::Cuda,
            det_size: 640,
            det_thresh: 0.5,
            nms: 0.4,
            flip_tta: true,
            threads: 8,
            max_faces: 4,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Tag {
    A,
    B,
    Live,
    /// reference photo loaded into the Webcam tab
    RefA,
    RefB,
}

#[derive(Clone)]
pub struct Entry {
    pub face: Face,
    pub emb: Vec<f32>,
    pub aligned: RgbImage,
    pub quality: Quality,
}

pub struct Analysis {
    pub entries: Vec<Entry>,
    pub ms: f32,
    pub det_ms: f32,
}

pub enum Job {
    Configure(Settings),
    Analyze {
        tag: Tag,
        seq: u64,
        img: RgbImage,
        fast: bool,
    },
    /// Part-based interpretability run on two already-detected faces.
    DeepAnalyze {
        a: Entry,
        b: Entry,
    },
    Stop,
}

pub enum Reply {
    Ready(String),
    Done {
        tag: Tag,
        seq: u64,
        analysis: Analysis,
    },
    Deep(Box<DeepReport>),
    Failed(String),
}

fn load(models: &Path, c: &Settings) -> Result<(Detector, Recognizer)> {
    let mut det = Detector::new(&models.join("det_10g.onnx"), c.provider, c.threads, c.det_size)?;
    let mut rec = Recognizer::new(&models.join("w600k_r50.onnx"), c.provider, c.threads)?;

    // Warm-up: force CUDA kernel init so the first real request is fast.
    let _ = det.detect(&RgbImage::new(c.det_size as u32, c.det_size as u32), 0.5, 0.4);
    let _ = rec.embed(&RgbImage::new(112, 112), false);
    Ok((det, rec))
}

pub fn spawn(models: PathBuf, initial: Settings) -> (Sender<Job>, Receiver<Reply>) {
    let (jtx, jrx) = channel::<Job>();
    let (rtx, rrx) = channel::<Reply>();

    std::thread::spawn(move || {
        let mut cfg = initial;

        let try_load = |cfg: &Settings, rtx: &Sender<Reply>| -> Option<(Detector, Recognizer)> {
            match load(&models, cfg) {
                Ok(p) => {
                    let _ = rtx.send(Reply::Ready(format!(
                        "Models ready - {} (requested)",
                        cfg.provider.label()
                    )));
                    Some(p)
                }
                Err(e) => {
                    let _ = rtx.send(Reply::Failed(format!("Failed to load models: {e:#}")));
                    None
                }
            }
        };

        let mut pair = try_load(&cfg, &rtx);

        while let Ok(job) = jrx.recv() {
            match job {
                Job::Stop => break,

                Job::Configure(s) => {
                    cfg = s;
                    pair = try_load(&cfg, &rtx);
                }

                Job::Analyze { tag, seq, img, fast } => {
                    let Some((det, rec)) = pair.as_mut() else {
                        let _ = rtx.send(Reply::Failed("models are not ready".into()));
                        continue;
                    };

                    let t0 = Instant::now();
                    let faces = match det.detect(&img, cfg.det_thresh, cfg.nms) {
                        Ok(f) => f,
                        Err(e) => {
                            let _ = rtx.send(Reply::Failed(format!("detection: {e:#}")));
                            continue;
                        }
                    };
                    let det_ms = t0.elapsed().as_secs_f32() * 1000.0;

                    let tta = cfg.flip_tta && !fast;
                    let mut entries = Vec::new();
                    for f in faces.iter().take(cfg.max_faces) {
                        let aligned = embed::align(&img, f);
                        match rec.embed(&aligned, tta) {
                            Ok(emb) => entries.push(Entry {
                                quality: analyze::assess(f, &aligned),
                                face: f.clone(),
                                emb,
                                aligned,
                            }),
                            Err(e) => {
                                let _ = rtx.send(Reply::Failed(format!("embedding: {e:#}")));
                            }
                        }
                    }

                    let ms = t0.elapsed().as_secs_f32() * 1000.0;
                    let _ = rtx.send(Reply::Done {
                        tag,
                        seq,
                        analysis: Analysis { entries, ms, det_ms },
                    });
                }

                Job::DeepAnalyze { a, b } => {
                    let Some((_, rec)) = pair.as_mut() else {
                        let _ = rtx.send(Reply::Failed("models are not ready".into()));
                        continue;
                    };

                    let t0 = Instant::now();
                    let mut forwards = 0usize;

                    // Re-compute a NO-TTA baseline so every delta is measured
                    // against the same preprocessing as the masked variants.
                    let base = match (rec.embed(&a.aligned, false), rec.embed(&b.aligned, false)) {
                        (Ok(x), Ok(y)) => Some((x, y)),
                        _ => None,
                    };
                    let Some((base_a, base_b)) = base else {
                        let _ = rtx.send(Reply::Failed("deep analysis: baseline failed".into()));
                        continue;
                    };
                    forwards += 2;
                    let cos_base = embed::cosine(&base_a, &base_b);

                    let mut scores: Vec<RegionScore> = Vec::new();
                    for r in regions::REGIONS {
                        let ao = regions::occlude(&a.aligned, &r.rect);
                        let bo = regions::occlude(&b.aligned, &r.rect);
                        let ai = regions::isolate(&a.aligned, &r.rect);
                        let bi = regions::isolate(&b.aligned, &r.rect);

                        let occ = match (rec.embed(&ao, false), rec.embed(&bo, false)) {
                            (Ok(x), Ok(y)) => Some(embed::cosine(&x, &y)),
                            _ => None,
                        };
                        let only = match (rec.embed(&ai, false), rec.embed(&bi, false)) {
                            (Ok(x), Ok(y)) => Some(embed::cosine(&x, &y)),
                            _ => None,
                        };
                        forwards += 4;

                        if let (Some(co), Some(cr)) = (occ, only) {
                            scores.push(RegionScore {
                                name: r.name,
                                note: r.note,
                                cos_occluded: co,
                                delta: cos_base - co,
                                cos_region_only: cr,
                            });
                        }
                    }

                    if scores.is_empty() {
                        let _ = rtx.send(Reply::Failed("deep analysis produced no regions".into()));
                        continue;
                    }

                    let report = DeepReport {
                        cos_headline: embed::cosine(&a.emb, &b.emb),
                        cos_base,
                        regions: scores,
                        geo_a: regions::geometry(&a.face),
                        geo_b: regions::geometry(&b.face),
                        aligned_a: a.aligned.clone(),
                        aligned_b: b.aligned.clone(),
                        heatmap: regions::diff_heatmap(&a.aligned, &b.aligned),
                        ms: t0.elapsed().as_secs_f32() * 1000.0,
                        forwards,
                    };
                    let _ = rtx.send(Reply::Deep(Box::new(report)));
                }
            }
        }
    });

    (jtx, rrx)
}