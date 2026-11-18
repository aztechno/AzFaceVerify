# Third-party notices

AzFaceVerify's own source code is MIT licensed (see `LICENSE`).
Everything below belongs to its respective authors and is **not** covered by
that license.

## Model weights (NOT included in this repository)

| Asset | Source | Terms |
|---|---|---|
| `det_10g.onnx` (SCRFD-10GF) | InsightFace `buffalo_l` model pack | InsightFace publishes its pretrained models for **non-commercial research** purposes. Read the current terms in the InsightFace repository before any commercial use. |
| `w600k_r50.onnx` (ArcFace ResNet50, WebFace600K) | InsightFace `buffalo_l` model pack | Same as above. |

You obtain these yourself by running `scripts/get-models.ps1`.
They are deliberately **not** committed to this repository, both because of
their license and because `w600k_r50.onnx` (~166 MB) exceeds GitHub's 100 MB
per-file limit.

If you need a permissively licensed pipeline, look at OpenCV Zoo
(YuNet + SFace) or train your own model. Note that swapping the recognition
model requires changing the pre/post-processing in `src/embed.rs`.

## Runtime binaries (NOT included in this repository)

| Component | Project | License |
|---|---|---|
| ONNX Runtime | microsoft/onnxruntime | MIT |
| CUDA / cuDNN | NVIDIA | NVIDIA proprietary EULA |
| DirectML | microsoft/DirectML | MIT (redistributable via Windows) |

`ort`'s `copy-dylibs` feature places the ONNX Runtime provider DLLs next to the
built executable. Those DLLs are not source code and are not committed here.

## Rust crates

Typical licenses are listed below. **Verify the exact set for your lockfile**
with `cargo install cargo-license && cargo license`.

| Crate | Typical license |
|---|---|
| `ort`, `ort-sys` | MIT OR Apache-2.0 |
| `eframe`, `egui`, `epaint`, `egui-winit` | MIT OR Apache-2.0 |
| `image` | MIT OR Apache-2.0 |
| `nokhwa` | MIT |
| `rfd` | MIT |
| `arboard` | MIT OR Apache-2.0 |
| `chrono` | MIT OR Apache-2.0 |
| `clap` | MIT OR Apache-2.0 |
| `serde`, `serde_json` | MIT OR Apache-2.0 |
| `anyhow` | MIT OR Apache-2.0 |
| `ndarray` | MIT OR Apache-2.0 |
| `tracing`, `tracing-subscriber` | MIT |
| `winresource` (build dependency) | MIT |

## Trademarks

NVIDIA, CUDA and GeForce are trademarks of NVIDIA Corporation.
Microsoft, Windows and DirectX are trademarks of Microsoft Corporation.
This project is not affiliated with or endorsed by InsightFace, NVIDIA or Microsoft.