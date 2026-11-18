//! Threshold and probability calibration from a labelled CSV of image pairs.

use crate::embed::{self, Recognizer};
use crate::engine::Provider;
use crate::scrfd::Detector;
use anyhow::{anyhow, Context, Result};
use std::path::Path;

fn embed_one(det: &mut Detector, rec: &mut Recognizer, p: &Path) -> Result<Vec<f32>> {
    let img = image::open(p).with_context(|| p.display().to_string())?.to_rgb8();
    let faces = det.detect(&img, 0.5, 0.4)?;
    let f = faces.first().ok_or_else(|| anyhow!("no face detected: {}", p.display()))?;
    rec.embed(&embed::align(&img, f), true)
}

pub fn run(csv: &Path, models: &Path, provider: Provider, out: &Path) -> Result<()> {
    let mut det = Detector::new(&models.join("det_10g.onnx"), provider, 8, 640)?;
    let mut rec = Recognizer::new(&models.join("w600k_r50.onnx"), provider, 8)?;

    let text = std::fs::read_to_string(csv)?;
    let (mut scores, mut labels) = (Vec::new(), Vec::new());

    for (i, line) in text.lines().enumerate().skip(1) {
        let c: Vec<&str> = line.split(',').map(|v| v.trim().trim_matches('"')).collect();
        if c.len() < 3 { continue; }
        let label: i32 = c[2].parse().unwrap_or(-1);
        if label != 0 && label != 1 { continue; }
        match (embed_one(&mut det, &mut rec, Path::new(c[0])), embed_one(&mut det, &mut rec, Path::new(c[1]))) {
            (Ok(a), Ok(b)) => { scores.push(embed::cosine(&a, &b)); labels.push(label); }
            (ea, eb) => eprintln!("row {} skipped: {:?} {:?}", i + 1, ea.err(), eb.err()),
        }
    }

    if scores.len() < 20 { return Err(anyhow!("too few valid pairs ({})", scores.len())); }

    let mut pos: Vec<f32> = scores.iter().zip(&labels).filter(|(_, &l)| l == 1).map(|(&v, _)| v).collect();
    let mut neg: Vec<f32> = scores.iter().zip(&labels).filter(|(_, &l)| l == 0).map(|(&v, _)| v).collect();
    if pos.is_empty() || neg.is_empty() { return Err(anyhow!("need both positive and negative samples")); }
    pos.sort_by(f32::total_cmp);
    neg.sort_by(f32::total_cmp);
    let (np, nn) = (pos.len() as f32, neg.len() as f32);

    let far_at = |t: f32| (neg.len() - neg.partition_point(|&v| v < t)) as f32 / nn;
    let tar_at = |t: f32| (pos.len() - pos.partition_point(|&v| v < t)) as f32 / np;

    let mut eer = (f32::MAX, 0.4f32, 0.0f32);
    for &t in &scores {
        let (far, frr) = (far_at(t), 1.0 - tar_at(t));
        if (far - frr).abs() < eer.0 { eer = ((far - frr).abs(), t, (far + frr) / 2.0); }
    }

    let thr_far = |target: f32| -> (f32, f32) {
        let k = ((target * nn).floor() as usize).min(neg.len() - 1);
        let t = neg[neg.len() - 1 - k];
        (t, tar_at(t))
    };
    let (t3, tar3) = thr_far(1e-3);
    let (t4, tar4) = thr_far(1e-4);

    let (mut w, mut c0) = (14.0f32, -5.6f32);
    let n = scores.len() as f32;
    for _ in 0..5000 {
        let (mut gw, mut gc) = (0.0f32, 0.0f32);
        for (&s, &y) in scores.iter().zip(&labels) {
            let p = 1.0 / (1.0 + (-(w * s + c0)).exp());
            let d = p - y as f32;
            gw += d * s;
            gc += d;
        }
        w -= 20.0 * (gw / n + 1e-3 * w);
        c0 -= 2.0 * (gc / n);
    }
    let slope = w.clamp(1.0, 60.0);
    let mid = -c0 / slope;
    let threshold = if neg.len() >= 1000 { t3 } else { eer.1 };

    println!("pairs            : {} positive, {} negative", pos.len(), neg.len());
    println!("EER              : {:.2}%  @ cosine {:.4}", eer.2 * 100.0, eer.1);
    println!("TAR @ FAR=1e-3   : {:.2}%  @ cosine {:.4}", tar3 * 100.0, t3);
    println!("TAR @ FAR=1e-4   : {:.2}%  @ cosine {:.4}", tar4 * 100.0, t4);
    println!("Platt fit        : slope={slope:.3} mid={mid:.4}");
    println!("chosen threshold : {threshold:.4}");
    if neg.len() < 10000 { println!("warning: fewer than 10k negatives - the FAR=1e-4 estimate is unstable."); }

    let json = serde_json::json!({
        "slope": slope, "mid": mid, "threshold": threshold,
        "eer": eer.2, "eer_threshold": eer.1,
        "tar_at_far_1e-3": tar3, "thr_far_1e-3": t3,
        "tar_at_far_1e-4": tar4, "thr_far_1e-4": t4,
        "n_pos": pos.len(), "n_neg": neg.len()
    });
    std::fs::write(out, serde_json::to_string_pretty(&json)?)?;
    println!("saved: {}", out.display());
    Ok(())
}