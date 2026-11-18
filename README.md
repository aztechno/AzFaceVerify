<div align="center">

# AzFaceVerify

**Offline 1:1 face verification for Windows, written in Rust.**

Compare two photos, or match a live webcam face against a reference photo.
Everything runs on your machine — no cloud, no network calls, no telemetry.

Built by **Azis Ichwan**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Platform](https://img.shields.io/badge/platform-Windows%2010%20%7C%2011%20x64-0078D6)
![Rust](https://img.shields.io/badge/Rust-stable-orange?logo=rust)
![Privacy](https://img.shields.io/badge/privacy-100%25%20local-success)

![screenshot](docs/screenshot.png)

</div>

---

## Table of contents

- [Features](#features)
- [How it works](#how-it-works)
- [Read this before trusting any number](#read-this-before-trusting-any-number)
- [Deep Analysis: which facial area matters?](#deep-analysis-which-facial-area-matters)
- [Known limitations](#known-limitations)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Usage](#usage)
- [Calibrating the probability](#calibrating-the-probability)
- [Portable build](#portable-build)
- [Performance](#performance)
- [Project structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Privacy and legal](#privacy-and-legal)
- [License](#license)
- [Roadmap](#roadmap)

---

## Features

| | |
|---|---|
| **Compare 2 photos** | JPG / PNG / BMP / WEBP, any mix of formats. Drag & drop, file picker, or paste straight from the clipboard. Rotate buttons for EXIF-rotated phone photos. |
| **Webcam recognition** | Load a reference **photo** and let the camera recognise against it live, or enrol a reference from 12 averaged webcam frames. Mix and match: A from a photo, B from the webcam. |
| **Deep Analysis** | Shows *which facial region* supports or contradicts the match — eyes, nose, mouth, jaw, forehead — using occlusion attribution. |
| **Honest scoring** | Raw cosine similarity, L2 distance, and a *calibrated* probability, with an explicit **INCONCLUSIVE** band instead of false confidence. |
| **Quality gate** | Face size, blur, brightness, contrast and head pose, with warnings when the input cannot be trusted. |
| **Multi-face support** | Click a bounding box to choose which face to compare. |
| **Timestamped reports** | `.txt` and `.json`, self-describing filenames, 512-d embeddings included in the JSON. |
| **GPU accelerated** | ONNX Runtime with **CUDA**, **DirectML**, or **CPU** fallback. |
| **Calibration tooling** | CLI that computes EER, TAR@FAR and fits a logistic mapping from your own labelled pairs. |
| **Portable output** | One script produces a clean `dist\` folder with launchers and a README. |

## How it works

```
image ──► SCRFD-10GF ──► bounding box + 5 landmarks ──► NMS
      ──► 5-point similarity alignment (Procrustes) ──► 112×112 crop
      ──► ArcFace ResNet50 (WebFace600K) ──► 512-d embedding ──► L2 normalise
      ──► cosine similarity ──► logistic calibration ──► verdict + quality report
```

| Stage | Model | Size | Source |
|---|---|---|---|
| Detection + landmarks | `det_10g.onnx` (SCRFD-10GF) | ~16 MB | InsightFace `buffalo_l` |
| Recognition | `w600k_r50.onnx` (ArcFace R50) | ~166 MB | InsightFace `buffalo_l` |

Models are **not** committed to this repository — see [Quick start](#quick-start).

Design notes:

- **All ORT API contact is isolated in `src/engine.rs`.** If you change the `ort`
  version, that is the only file you should need to touch.
- **Inference runs on a dedicated worker thread.** The GUI never blocks; the
  webcam path uses back-pressure (a new frame is only queued when the worker is idle).
- **Alignment is implemented from scratch** (closed-form 2D Procrustes + bilinear
  warp), so there is no OpenCV dependency.

---

## Read this before trusting any number

**Cosine similarity is not a probability.** A cosine of `0.60` does *not* mean
"60% similar". AzFaceVerify therefore shows three separate things:

1. **Raw cosine** (−1 … 1) — the model's actual output, auditable.
2. **Calibrated probability** — `p = sigmoid(slope × (cos − mid))`.
   With the shipped defaults (`slope = 14`, `mid = 0.40`) this is a **heuristic**.
   Run the [calibration command](#calibrating-the-probability) on your own labelled
   pairs to make it statistically meaningful.
3. **Verdict band** — with an explicit grey zone, because saying *"retake the photo"*
   is better than being confidently wrong.

With the default threshold `t = 0.40`:

| Condition | Verdict |
|---|---|
| `cos ≥ t + 0.10` | VERY LIKELY THE SAME PERSON |
| `cos ≥ t` | LIKELY THE SAME PERSON |
| `cos ≥ t − 0.08` | **INCONCLUSIVE — retake the photos** |
| `cos ≥ t − 0.20` | LIKELY DIFFERENT PEOPLE |
| otherwise | VERY LIKELY DIFFERENT PEOPLE |

> Do not use a single score as the sole basis for a high-stakes decision.

---

## Deep Analysis: which facial area matters?

ArcFace outputs **one global 512-d vector**. There is no per-organ similarity
output, and any tool claiming otherwise is inventing it. AzFaceVerify derives
region evidence instead.

### Occlusion attribution (the defensible method)

The same region is hidden with neutral grey in **both** aligned faces, the model
runs again, and the change in cosine is measured:

```
delta = baseline_cosine − cosine_with_region_hidden

delta > 0  →  that region was evidence FOR "same person"
delta < 0  →  that region was pushing TOWARD "different"
```

Deltas are measured against a **no-TTA baseline** so the comparison is
apples-to-apples. A full run is **22 model forwards** (2 baseline + 5 regions × 4).

Example reading:

| Region | Δ | Interpretation |
|---|---:|---|
| Eyes + eyebrows | **+0.21** | strongest evidence for *same person* |
| Nose | +0.06 | supports the match |
| Forehead + hairline | +0.01 | negligible — a hat would not hurt much |
| Chin + jawline | −0.04 | mild disagreement, likely beard growth |
| Mouth | **−0.11** | main disagreement — likely expression, not identity |

> *"Eyes and nose agree; the mouth differs because one photo is laughing —
> that lowers the score even though the identity is the same."*
> That insight is invisible in a single cosine number.

### Regions

| Region | What usually affects it |
|---|---|
| Eyes + eyebrows | Strongest identity carrier. Sunglasses / fringe hurt the most. |
| Nose | Stable across expression and facial hair, sensitive to yaw. |
| Mouth | Expression, smiling, moustache, lipstick. |
| Chin + jawline | Beard growth, weight change, head pitch. |
| Forehead + hairline | Hats, caps, hijab, fringe. Usually a weak contributor. |

### Also shown, clearly labelled as weaker

- **Region-only cosine** — only that region stays visible. The model was trained on
  *full* faces, so these embeddings are **out-of-distribution**. Use for **ranking
  regions only**; never compare them to the full-face threshold.
- **Landmark geometry** — 8 ratios from 5 landmarks, normalised by inter-ocular
  distance. Dominated by pose and expression. Not a morphometric measurement.
- **Difference heatmap** — visual only. Dominated by lighting, pose and expression.
  **Not identity evidence.**

Caveats: masking itself shifts the input distribution, regions overlap, and deltas
do **not** sum to the total score.

---

## Known limitations

| Situation | Behaviour |
|---|---|
| Glasses, moustache, beard, hat / head covering, makeup | Generally handled well; score drops slightly |
| Face mask, large dark sunglasses | **Significant degradation** — needs a periocular model |
| Identical twins / strong look-alikes | **Hardest case.** No threshold fixes this |
| Photo or video held up to the webcam | **Will pass.** No liveness detection is implemented |
| Face shorter side < 60 px, heavy blur, extreme pose | Unreliable — the app warns you |
| Demographic variation | All face models carry some bias; measure on your own population |

The app also cannot detect occlusion explicitly. Image quality and pose metrics act
as proxies, and are labelled as heuristics.

---

## Requirements

- **Windows 10 / 11 x64**
- **Rust** (MSVC toolchain) + **Visual Studio Build Tools 2022** with the
  *"Desktop development with C++"* workload and a Windows SDK
- **CUDA mode:** CUDA 12.x + cuDNN 9.x reachable on `PATH`
- **DirectML mode:** nothing extra — any DirectX 12 GPU
- **CPU mode:** nothing extra

Reference machine: **Ryzen 7 5800H · 24 GB RAM · RTX 3060 Laptop**.

---

## Quick start

```powershell
git clone https://github.com/aztechno/AzFaceVerify.git
cd AzFaceVerify

# 1. Download the two ONNX models (~275 MB archive)
powershell -ExecutionPolicy Bypass -File scripts\get-models.ps1

# 2. Build and run (always --release)
cargo run --release                          # tries CUDA, falls back to CPU
cargo run --release -- --provider directml   # no CUDA install needed
cargo run --release -- --provider cpu
```

> **Always use `--release`.** Debug builds are dramatically slower in the
> warp / normalise path.

### Why are the models not in the repo?

Two hard reasons:

1. `w600k_r50.onnx` is ~166 MB — **GitHub rejects files over 100 MB.**
2. InsightFace publishes its pretrained weights for **non-commercial research**
   use. Redistributing them here would be a licensing problem.

`scripts\get-models.ps1` fetches them for you. `.gitignore` excludes `models/`,
`target/` and `dist/`.

### Making CUDA work

ONNX Runtime 1.20+ needs **CUDA 12.x + cuDNN 9.x**. The lightest route:

```powershell
pip install nvidia-cudnn-cu12 nvidia-cublas-cu12 nvidia-cufft-cu12 nvidia-curand-cu12
# then add each ...\site-packages\nvidia\<lib>\bin folder to PATH
```

Or install the official CUDA Toolkit 12.x + cuDNN 9.x and copy the DLLs next to
the executable.

> **The status bar shows the *requested* provider.** If CUDA/cuDNN is missing,
> ONNX Runtime silently falls back to CPU. Confirm real GPU usage in
> **Task Manager → Performance → GPU**, or run with `RUST_LOG=ort=debug`.
> If CUDA is a hassle, just use **DirectML** — on models this small the
> difference is modest and it needs zero extra installs.

---

## Usage

### GUI

| Tab | What you do there |
|---|---|
| **Compare 2 Photos** | Load A and B (drag & drop / Open / Paste). Click a box to pick a face. The verdict is pinned at the bottom — drag its top edge to resize. |
| **Webcam** | `Load photo → A/B` to use a still image as reference, or `Record A/B from webcam` to average 12 good frames. Then pick `Live match against: A/B`. |
| **Deep Analysis** | Run occlusion attribution on the photo pair or the webcam reference pair. |
| **Settings** | Provider, detector input size, threads, flip-TTA, thresholds, preview height, calibration load/reset. |

Quality readout per face: size in px, detector confidence, blur (variance of
Laplacian), brightness, contrast, and yaw / roll / pitch estimates — each with
warnings when it falls outside a sane range.

### CLI

```powershell
# quick check without opening the GUI
AzFaceVerify.exe compare a.jpg b.png

# fit a threshold + probability mapping from labelled pairs
AzFaceVerify.exe calibrate pairs.csv --out calibration.json

# force a provider / model folder
AzFaceVerify.exe --provider directml --models models
```

If `models\` is not found relative to the working directory, the app also looks
next to the executable — so a portable folder works when launched from a shortcut.

---

## Calibrating the probability

Collect pairs from **your own** camera, lighting and distance. Aim for 500+
positive and 2,000–10,000 negative pairs, including glasses / beard / hat / makeup
variation.

`pairs.csv`:

```csv
path_a,path_b,label
data/ali_1.jpg,data/ali_2.jpg,1
data/ali_1.jpg,data/budi_1.jpg,0
```

`1` = same person, `0` = different people.

```powershell
AzFaceVerify.exe calibrate pairs.csv --out calibration.json
```

Outputs **EER**, **TAR@FAR = 1e-3**, **TAR@FAR = 1e-4**, and a fitted logistic
mapping (`slope`, `mid`, `threshold`). Load the JSON in
**Settings → Decision & calibration**.

> Public benchmark numbers (e.g. ~99.8% on LFW for this model class) do **not**
> describe your deployment conditions. Measure your own.
> Fewer than ~10,000 negatives makes the FAR = 1e-4 estimate unstable, and the
> tool will tell you so.

---

## Portable build

```powershell
powershell -ExecutionPolicy Bypass -File scripts\make-dist.ps1 -Zip
powershell -ExecutionPolicy Bypass -File scripts\make-dist.ps1 -NoCuda -Zip
```

Produces `dist\AzFaceVerify\` containing:

```
AzFaceVerify.exe
*.dll                    ONNX Runtime providers + DirectML
models\det_10g.onnx
models\w600k_r50.onnx
assets\icon.ico
Run-CUDA.bat  Run-DirectML.bat  Run-CPU.bat
README.txt
```

Excluded on purpose: `.pdb`, `deps\`, `build\`, `incremental\`, `.fingerprint\`.

Options: `-NoCuda` drops the ~176 MB CUDA provider (DirectML/CPU only),
`-CudaBin <path>` bundles CUDA/cuDNN runtime DLLs for machines without CUDA
installed (expect 1–2 GB), `-SkipBuild` reuses the current build.

---

## Performance

Observed on the reference machine (Ryzen 7 5800H + RTX 3060 Laptop, CUDA provider),
single photo end-to-end including detection: **~215–230 ms** on early requests
after warm-up. Steady-state requests are typically faster; the first CUDA request
additionally pays kernel initialisation, which the app pre-warms at load.

The GUI displays **per-request timing** (total ms and detection ms), and Deep
Analysis reports its own runtime and forward count — **measure on your own
hardware** rather than trusting any table, including this one.

Rough guidance: DirectML is usually close to CUDA for models this small, and CPU
(8 threads) is usable but noticeably slower. VRAM footprint is well under 1.5 GB.

---

## Project structure

```
AzFaceVerify/
├─ Cargo.toml            ort and ort-sys are BOTH pinned to =2.0.0-rc.10
├─ Cargo.lock            committed on purpose (binary crate)
├─ build.rs              embeds icon + version info into the .exe (best-effort)
├─ assets/icon.ico       window / taskbar / exe icon
├─ docs/screenshot.png
├─ scripts/
│  ├─ get-models.ps1     download + extract the two ONNX files
│  ├─ make-dist.ps1      build the portable folder
│  └─ publish-github.ps1 init + commit + push, with a >95 MB safety guard
└─ src/
   ├─ main.rs            CLI entry, model-path resolution
   ├─ engine.rs          the ONLY file touching the ort tensor API
   ├─ scrfd.rs           SCRFD preprocessing, anchor decode, NMS
   ├─ embed.rs           Procrustes alignment, bilinear warp, ArcFace, flip-TTA
   ├─ analyze.rs         quality metrics, verdict bands, calibration math
   ├─ regions.rs         occlusion attribution, geometry, heatmap
   ├─ worker.rs          background inference thread (mpsc jobs/replies)
   ├─ cam.rs             webcam capture via Media Foundation
   ├─ calib.rs           EER, TAR@FAR, Platt scaling
   ├─ icon.rs            icon lookup + procedural fallback
   └─ app.rs             egui GUI
```

---

## Troubleshooting

These are real failures encountered while building this project.

| Symptom | Cause and fix |
|---|---|
| **Hundreds of errors inside `ort`**: `no method named unwrap_or_else found for fn pointer ...`, `expected "system" fn, found "C" fn` | **`ort` and `ort-sys` versions mismatch.** Cargo resolved `ort` rc.9 against `ort-sys` rc.10, which removed the `Option` wrappers from the C API struct. **Both are pinned to `=2.0.0-rc.10` in `Cargo.toml` — never unpin just one.** Then `cargo clean -p azfaceverify` and rebuild. |
| `cannot find EnvFilter in tracing_subscriber` / `no method with_env_filter` | `tracing-subscriber` needs `features = ["env-filter"]`. |
| `the ? operator cannot be applied to Vec<(Cow<str>, SessionInputValue)>` | Your `ort` RC returns a plain `Vec` from `ort::inputs!`. Remove that `?` in `engine.rs`. |
| `cannot borrow *session as mutable` | `ort` 2.x `Session::run()` requires `&mut Session`. `Detector::detect` and `Recognizer::embed` take `&mut self` accordingly. |
| `fn(_) -> DisplayValue<_> {display} doesn't implement Display` | A local variable named `display` is shadowed by `tracing::field::display` inside the macro expansion. Rename it. Avoid `display`, `debug`, `message` as variable names used inside `tracing` macros. |
| `cannot use self.seq because it was mutably borrowed` | Read `self.seq` into a local **before** taking the mutable slot borrow, and scope that borrow. |
| `cannot assign to self.status, which is behind a & reference` | The method needs `&mut self`; clone the borrowed data first so the immutable borrow ends. |
| warning: `falling back to f32 as f32: From<f64> is not satisfied` | Write `2.0_f32` explicitly in `Stroke::new` / `rect_stroke`. |
| `onnxruntime.dll not found` | Keep the `copy-dylibs` feature, or use `load-dynamic` + `ORT_DYLIB_PATH`. |
| CUDA silently falls back to CPU | `cudnn64_9.dll` / `cublas64_12.dll` not on `PATH`. Check Task Manager → GPU, or `RUST_LOG=ort=debug`. |
| No face detected in a phone photo | EXIF rotation. Use **Rotate L / Rotate R**. |
| `winresource` fails in `build.rs` | Delete `build.rs` and the `[build-dependencies]` block. The build continues and the **window** icon still works (loaded at runtime); only the Explorer file icon reverts. |
| `Copy-Item: Cannot overwrite the item ... with itself` | Old `get-models.ps1`. The current script checks `Test-Path` and only searches the extracted subfolder. |
| GUI: analysis text pushed out of view | Fixed by design: preview height is capped (Settings → Layout), the photo area scrolls, and the result panel is pinned at the bottom. |

**Settings → About → "Window icon: ..."** tells you whether your `icon.ico` was
actually loaded or the procedural fallback was used.

---

## Privacy and legal

- All processing is **local**. The application makes **no network calls**.
  (The only download is `get-models.ps1`, which you run yourself.)
- Face data is **biometric / special-category personal data**. In Indonesia this is
  *data pribadi spesifik* under **UU PDP No. 27/2022**.
- Obtain consent, define a retention policy, and **never store embeddings
  unencrypted**.
- **No liveness detection** — do not use this for access control or attendance
  without adding anti-spoofing.
- Do not make high-stakes decisions from a single score.

## License

**Source code: MIT** — see [`LICENSE`](LICENSE). Copyright (c) 2026 Azis Ichwan.

That license covers the code **only**. Model weights and runtime binaries are
**not included** and **not covered** by it — see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). In particular, InsightFace
pretrained models are published for **non-commercial research** use; read their
current terms before any commercial deployment.

## Roadmap

- [ ] FP16 / TensorRT execution provider
- [ ] Liveness / anti-spoofing (MiniFASNet or challenge-response)
- [ ] Two-model ensemble (average **scores**, never embeddings from different spaces)
- [ ] Periocular model for masked / sunglasses cases
- [ ] 1:N identification with an encrypted gallery
- [ ] Webcam face tracking so detection does not run every frame
- [ ] Automatic EXIF orientation handling
- [ ] Encrypted embedding storage (Windows DPAPI / AES-GCM)

## Acknowledgements

[InsightFace](https://github.com/deepinsight/insightface) (SCRFD + ArcFace),
[ONNX Runtime](https://github.com/microsoft/onnxruntime),
[`ort`](https://github.com/pykeio/ort),
[egui / eframe](https://github.com/emilk/egui),
[nokhwa](https://github.com/l1npengtul/nokhwa).

Not affiliated with or endorsed by InsightFace, NVIDIA or Microsoft.

---

<div align="center">

**Azis Ichwan** · [@aztechno](https://github.com/aztechno)

</div>