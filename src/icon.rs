//! AzFaceVerify application icon (window + taskbar).
//! Built by Azis Ichwan.
//!
//! Search order for the icon file:
//!   1. next to the executable        (exe_dir/assets/icon.ico, exe_dir/icon.ico, ...)
//!   2. current working directory     (./assets/icon.ico, ./icon.ico, ...)
//!   3. the cargo project directory   (compile-time CARGO_MANIFEST_DIR)
//!
//! If nothing is found, a small mark is generated procedurally so the window
//! never shows the default egui "e".
//!
//! NOTE: never name a local variable `display`, `debug` or `message` if it is
//! used inside a `tracing` macro - those macros import helpers with the same
//! names into the expansion scope and would shadow your variable.

use eframe::egui;
use image::imageops::FilterType;

const CANDIDATES: &[&str] = &[
    "assets/icon.ico",
    "icon.ico",
    "assets/icon.png",
    "icon.png",
];

/// Returns the icon plus a human-readable description of where it came from.
pub fn load() -> (egui::IconData, String) {
    match load_from_disk() {
        Some((icon, source)) => (icon, format!("loaded from {source}")),
        None => (
            generated(),
            "generated fallback (icon.ico not found)".to_string(),
        ),
    }
}

fn load_from_disk() -> Option<(egui::IconData, String)> {
    let mut roots: Vec<std::path::PathBuf> = Vec::new();

    if let Ok(exe) = std::env::current_exe() {
        if let Some(dir) = exe.parent() {
            roots.push(dir.to_path_buf());
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        roots.push(cwd);
    }
    roots.push(std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")));

    for root in roots {
        for name in CANDIDATES {
            let path = root.join(name);
            if !path.is_file() {
                continue;
            }
            let Ok(bytes) = std::fs::read(&path) else {
                continue;
            };
            let Ok(img) = image::load_from_memory(&bytes) else {
                continue;
            };

            // 64x64 RGBA is a good size for both the title bar and the taskbar.
            let rgba = image::imageops::resize(&img.to_rgba8(), 64, 64, FilterType::Lanczos3);
            let (width, height) = rgba.dimensions();

            // Renamed from `display` to `icon_path` to avoid the tracing macro
            // shadowing `tracing::field::display`.
            let icon_path = path.display().to_string();
            tracing::info!("AzFaceVerify: window icon loaded from {}", icon_path);

            return Some((
                egui::IconData {
                    rgba: rgba.into_raw(),
                    width,
                    height,
                },
                icon_path,
            ));
        }
    }
    None
}

/// Procedural fallback: dark rounded square, green face-scan brackets,
/// two eyes and a smile. Guarantees we never show the egui default icon.
fn generated() -> egui::IconData {
    const S: usize = 64;
    let mut buf = vec![0u8; S * S * 4];

    let bg = [18u8, 22, 30];
    let fg = [70u8, 220, 140];
    let white = [235u8, 240, 245];

    for y in 0..S {
        for x in 0..S {
            let fx = x as f32 + 0.5;
            let fy = y as f32 + 0.5;

            // Rounded-rectangle mask (transparent corners).
            let r = 12.0f32;
            let dx = (fx - 32.0).abs() - (32.0 - r);
            let dy = (fy - 32.0).abs() - (32.0 - r);
            let outside = (dx.max(0.0).powi(2) + dy.max(0.0).powi(2)).sqrt() - r;
            if outside > 0.0 {
                continue;
            }

            let mut c = bg;

            // Four corner brackets of a face-scan frame.
            let in_box = (14.0..=50.0).contains(&fx) && (14.0..=50.0).contains(&fy);
            if in_box {
                let near_x = fx < 21.0 || fx > 43.0;
                let near_y = fy < 21.0 || fy > 43.0;
                let edge_x = (fx - 14.0).abs() < 2.5 || (fx - 50.0).abs() < 2.5;
                let edge_y = (fy - 14.0).abs() < 2.5 || (fy - 50.0).abs() < 2.5;
                if (edge_x && near_y) || (edge_y && near_x) {
                    c = fg;
                }
            }

            // Eyes.
            for (ex, ey) in [(26.0f32, 28.0f32), (38.0f32, 28.0f32)] {
                if ((fx - ex).powi(2) + (fy - ey).powi(2)).sqrt() < 2.6 {
                    c = white;
                }
            }

            // Smile arc.
            let d = ((fx - 32.0).powi(2) + (fy - 33.0).powi(2)).sqrt();
            if (d - 9.0).abs() < 1.6 && fy > 36.0 {
                c = white;
            }

            let i = (y * S + x) * 4;
            buf[i] = c[0];
            buf[i + 1] = c[1];
            buf[i + 2] = c[2];
            buf[i + 3] = 255;
        }
    }

    egui::IconData {
        rgba: buf,
        width: S as u32,
        height: S as u32,
    }
}