# AzFaceVerify - portable distribution builder
# Built by Azis Ichwan
#
# Usage:
#   powershell -ExecutionPolicy Bypass -File scripts\make-dist.ps1
#   ... -NoCuda            (skip the 176 MB CUDA provider -> DirectML/CPU only)
#   ... -Zip               (also produce dist\AzFaceVerify.zip)
#   ... -CudaBin "C:\...\nvidia\cudnn\bin"   (bundle CUDA runtime DLLs)
#   ... -SkipBuild         (use the existing target\release build)

param(
    [string]$OutDir = "dist\AzFaceVerify",
    [string]$CudaBin = "",
    [switch]$NoCuda,
    [switch]$Zip,
    [switch]$SkipBuild
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

Write-Host "=== AzFaceVerify distribution builder ===" -ForegroundColor Cyan

if (-not $SkipBuild) {
    Write-Host "Building release..."
    cargo build --release
    if ($LASTEXITCODE -ne 0) { throw "cargo build failed" }
}

$rel = "target\release"
$exe = Join-Path $rel "AzFaceVerify.exe"
if (-not (Test-Path $exe)) { throw "$exe not found. Build first." }

if (Test-Path $OutDir) {
    Write-Host "Cleaning $OutDir ..."
    Remove-Item $OutDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
New-Item -ItemType Directory -Force -Path "$OutDir\models" | Out-Null
New-Item -ItemType Directory -Force -Path "$OutDir\assets" | Out-Null

# 1. Executable (NOT the .pdb - that is 23 MB of debug symbols)
Copy-Item $exe $OutDir -Force
Write-Host "  + AzFaceVerify.exe"

# 2. Runtime DLLs that sit next to the exe (top level only, never deps\)
Get-ChildItem "$rel\*.dll" -File | ForEach-Object {
    if ($NoCuda -and $_.Name -like "*providers_cuda*") {
        Write-Host "  - skipped $($_.Name) (-NoCuda)"
    } else {
        Copy-Item $_.FullName $OutDir -Force
        Write-Host "  + $($_.Name)"
    }
}

# 3. Only the two models the app actually loads
foreach ($m in 'det_10g.onnx', 'w600k_r50.onnx') {
    $src = "models\$m"
    if (-not (Test-Path $src)) { throw "$src missing. Run scripts\get-models.ps1 first." }
    Copy-Item $src "$OutDir\models\$m" -Force
    Write-Host "  + models\$m"
}

# 4. Window icon (the exe already embeds it, this is for the runtime loader)
if (Test-Path "assets\icon.ico") {
    Copy-Item "assets\icon.ico" "$OutDir\assets\icon.ico" -Force
    Write-Host "  + assets\icon.ico"
} elseif (Test-Path "icon.ico") {
    Copy-Item "icon.ico" "$OutDir\assets\icon.ico" -Force
    Write-Host "  + assets\icon.ico (from root)"
}

# 5. Optional: bundle CUDA / cuDNN runtime so it runs on PCs without CUDA installed
if ($CudaBin -ne "") {
    if (-not (Test-Path $CudaBin)) { throw "CudaBin not found: $CudaBin" }
    Write-Host "Bundling CUDA runtime from $CudaBin (this can be 1-2 GB)..."
    $patterns = @(
        'cudart64_*.dll', 'cublas64_*.dll', 'cublasLt64_*.dll',
        'cufft64_*.dll', 'curand64_*.dll', 'cudnn*64_*.dll',
        'nvrtc64_*.dll', 'nvrtc-builtins64_*.dll', 'zlibwapi.dll'
    )
    foreach ($p in $patterns) {
        Get-ChildItem -Path $CudaBin -Filter $p -File -ErrorAction SilentlyContinue | ForEach-Object {
            Copy-Item $_.FullName $OutDir -Force
            Write-Host "  + $($_.Name)"
        }
    }
}

# 6. Launchers
@"
@echo off
cd /d "%~dp0"
start "" "AzFaceVerify.exe" --provider cuda --models models
"@ | Set-Content "$OutDir\Run-CUDA.bat" -Encoding ASCII

@"
@echo off
cd /d "%~dp0"
start "" "AzFaceVerify.exe" --provider directml --models models
"@ | Set-Content "$OutDir\Run-DirectML.bat" -Encoding ASCII

@"
@echo off
cd /d "%~dp0"
start "" "AzFaceVerify.exe" --provider cpu --models models
"@ | Set-Content "$OutDir\Run-CPU.bat" -Encoding ASCII

# 7. README
@"
AzFaceVerify - offline 1:1 face verification
Built by Azis Ichwan

HOW TO RUN
  Double-click AzFaceVerify.exe          (tries CUDA, falls back to CPU)
  Run-CUDA.bat                           NVIDIA GPU + CUDA 12.x + cuDNN 9.x
  Run-DirectML.bat                       any DirectX 12 GPU, no CUDA install needed
  Run-CPU.bat                            processor only

REQUIREMENTS
  Windows 10/11 x64.
  CUDA mode additionally needs CUDA 12.x + cuDNN 9.x, either installed on the
  target PC or bundled next to this exe. If they are missing, ONNX Runtime
  silently falls back to CPU - the status bar shows the REQUESTED provider,
  so verify actual GPU usage in Task Manager.
  DirectML mode needs nothing extra.

WHAT IS IN HERE
  AzFaceVerify.exe                  the application
  *.dll                             ONNX Runtime execution providers + DirectML
  models\det_10g.onnx               SCRFD face detector
  models\w600k_r50.onnx             ArcFace R50 recognition model
  assets\icon.ico                   window / taskbar icon

COMMAND LINE
  AzFaceVerify.exe compare a.jpg b.png
  AzFaceVerify.exe calibrate pairs.csv --out calibration.json
  AzFaceVerify.exe --provider directml --models models

IMPORTANT LIMITATIONS
  - Cosine similarity is a raw model score, not a probability. The shown
    percentage is a logistic calibration; run 'calibrate' on your own labelled
    pairs to make it statistically meaningful.
  - The Deep Analysis tab uses occlusion attribution. It shows which facial
    region supports or contradicts the match. It is NOT a per-organ similarity
    readout from the model.
  - Glasses, moustache, beard, hats and makeup are generally handled well.
    Face masks, dark sunglasses and identical twins are known failure modes.
  - No liveness detection: a photo held up to the webcam will pass.

LICENSE / PRIVACY
  InsightFace pretrained weights are licensed for NON-COMMERCIAL RESEARCH.
  Check the model license before any commercial use.
  All processing is local; the app makes no network calls.
  Face data is biometric / special-category personal data. Obtain consent,
  define a retention policy, and never store embeddings unencrypted.
"@ | Set-Content "$OutDir\README.txt" -Encoding UTF8

Write-Host "  + README.txt, Run-CUDA.bat, Run-DirectML.bat, Run-CPU.bat"

# 8. Optional zip
if ($Zip) {
    $zipPath = "dist\AzFaceVerify.zip"
    if (Test-Path $zipPath) { Remove-Item $zipPath -Force }
    Write-Host "Creating $zipPath ..."
    Compress-Archive -Path $OutDir -DestinationPath $zipPath
}

$size = (Get-ChildItem $OutDir -Recurse -File | Measure-Object -Property Length -Sum).Sum
Write-Host ""
Write-Host ("Done: {0}  ({1:N1} MB)" -f $OutDir, ($size / 1MB)) -ForegroundColor Green
if ($Zip) {
    $zs = (Get-Item "dist\AzFaceVerify.zip").Length
    Write-Host ("Zip : dist\AzFaceVerify.zip  ({0:N1} MB)" -f ($zs / 1MB)) -ForegroundColor Green
}
Write-Host ""
Write-Host "Excluded on purpose: .pdb, deps\, build\, incremental\, .fingerprint\, examples\, lock files."