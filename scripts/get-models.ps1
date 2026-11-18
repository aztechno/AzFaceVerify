# AzFaceVerify - model downloader
# Built by Azis Ichwan
# Downloads InsightFace buffalo_l and extracts the two ONNX files we need.

$ErrorActionPreference = 'Stop'

New-Item -ItemType Directory -Force -Path models | Out-Null

$url = 'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip'
$zip = 'models\buffalo_l.zip'

if (-not (Test-Path $zip)) {
    Write-Host "Downloading buffalo_l (~275 MB)..."
    Invoke-WebRequest $url -OutFile $zip
} else {
    Write-Host "Zip already downloaded, skipping download."
}

Write-Host "Extracting..."
Expand-Archive $zip -DestinationPath models\buffalo_l -Force

foreach ($name in 'det_10g.onnx', 'w600k_r50.onnx') {
    $dest = "models\$name"
    
    # Skip if file already successfully copied to the root models folder
    if (Test-Path $dest) {
        Write-Host "Already exists: $dest"
        continue
    }

    # Search only inside the extracted subfolder to avoid copying a file onto itself
    $found = Get-ChildItem models\buffalo_l -Recurse -Filter $name -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $found) { 
        throw "$name was not found inside the extracted archive" 
    }
    
    Copy-Item $found.FullName $dest -Force
    Write-Host "OK: $dest"
}

Write-Host ""
Write-Host "Done. Note: InsightFace pretrained weights are licensed for non-commercial research use."