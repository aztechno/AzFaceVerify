//! AzFaceVerify - offline 1:1 face verification.
//! Built by Azis Ichwan.

#![windows_subsystem = "windows"]

mod analyze;
mod app;
mod calib;
mod cam;
mod embed;
mod engine;
mod icon;
mod regions;
mod scrfd;
mod worker;

use clap::{Parser, Subcommand};
use std::path::PathBuf;

pub const APP_NAME: &str = "AzFaceVerify";
pub const APP_AUTHOR: &str = "Azis Ichwan";
pub const APP_VERSION: &str = env!("CARGO_PKG_VERSION");

#[derive(Parser)]
#[command(
    name = "AzFaceVerify",
    version,
    author = "Azis Ichwan",
    about = "Offline 1:1 face verification (Rust + ONNX Runtime + CUDA)"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Option<Cmd>,

    /// Folder that contains det_10g.onnx and w600k_r50.onnx
    #[arg(long, default_value = "models")]
    models: PathBuf,

    /// Execution provider: cuda | directml | cpu
    #[arg(long, default_value = "cuda")]
    provider: String,
}

#[derive(Subcommand)]
enum Cmd {
    /// Launch the graphical interface (default)
    Gui,
    /// Quick comparison of two images without the GUI
    Compare { a: PathBuf, b: PathBuf },
    /// Calibrate threshold and probability mapping from a labelled CSV
    Calibrate {
        csv: PathBuf,
        #[arg(long, default_value = "calibration.json")]
        out: PathBuf,
    },
}

/// Make the portable "dist" folder work when the exe is started from a
/// shortcut or from another working directory: fall back to <exe_dir>/models.
fn resolve_models(requested: PathBuf) -> PathBuf {
    if requested.join("det_10g.onnx").is_file() {
        return requested;
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            let a = dir.join(&requested);
            if a.join("det_10g.onnx").is_file() {
                return a;
            }
            let b = dir.join("models");
            if b.join("det_10g.onnx").is_file() {
                return b;
            }
        }
    }
    requested
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "warn,ort=info".into()),
        )
        .init();

    let cli = Cli::parse();
    let provider = engine::Provider::parse(&cli.provider);
    let models = resolve_models(cli.models);

    println!("{APP_NAME} v{APP_VERSION} - built by {APP_AUTHOR}");
    println!("models folder : {}", models.display());
    println!("provider      : {}", provider.label());

    match cli.cmd.unwrap_or(Cmd::Gui) {
        Cmd::Gui => app::run(models, provider),

        Cmd::Calibrate { csv, out } => calib::run(&csv, &models, provider, &out),

        Cmd::Compare { a, b } => {
            let mut det = scrfd::Detector::new(&models.join("det_10g.onnx"), provider, 8, 640)?;
            let mut rec = embed::Recognizer::new(&models.join("w600k_r50.onnx"), provider, 8)?;

            let process = |p: &std::path::Path,
                           det: &mut scrfd::Detector,
                           rec: &mut embed::Recognizer|
             -> anyhow::Result<Vec<f32>> {
                let img = image::open(p)?.to_rgb8();
                let faces = det.detect(&img, 0.5, 0.4)?;
                let f = faces
                    .first()
                    .ok_or_else(|| anyhow::anyhow!("no face detected in {}", p.display()))?;
                rec.embed(&embed::align(&img, f), true)
            };

            let emb_a = process(&a, &mut det, &mut rec)?;
            let emb_b = process(&b, &mut det, &mut rec)?;

            let v = analyze::verdict(&emb_a, &emb_b, &analyze::Calibration::default());
            println!();
            println!("verdict : {}", v.label);
            println!("cosine  : {:.4}", v.cos);
            println!("l2      : {:.4}", v.l2);
            println!(
                "prob    : {:.1}%  (DEFAULT calibration - run `calibrate` for meaningful numbers)",
                v.prob * 100.0
            );
            Ok(())
        }
    }
}