<#
    build_msix.ps1 — Build a MotionBloom MSIX on Windows.

    Prereqs (install once):
      - Python 3.11 x64       (winget install Python.Python.3.11)
      - Windows 10/11 SDK     (for MakeAppx.exe / SignTool.exe)
      - Git (optional)

    Usage (from repo root, in an elevated PowerShell):
      powershell -ExecutionPolicy Bypass -File packaging\build_msix.ps1

    Output:
      dist\MotionBloom\               (PyInstaller one-dir build)
      dist\MotionBloom.msix           (final package to upload to Partner Center)
#>

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

# Move to repo root (parent of this script's folder).
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo
Write-Host "Repo root: $repo"

# --- 1. Python venv + deps ------------------------------------------------
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    py -3.11 -m venv .venv
}
& .\.venv\Scripts\python.exe -m pip install --upgrade pip wheel | Out-Null
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt
& .\.venv\Scripts\python.exe -m pip install pyinstaller

# --- 2. PyInstaller freeze ------------------------------------------------
if (Test-Path "dist\MotionBloom")   { Remove-Item -Recurse -Force "dist\MotionBloom" }
if (Test-Path "dist\MotionBloom.msix") { Remove-Item -Force "dist\MotionBloom.msix" }
if (Test-Path "build")              { Remove-Item -Recurse -Force "build" }

Write-Host "Running PyInstaller..."
& .\.venv\Scripts\pyinstaller.exe packaging\motionbloom.spec --noconfirm

$appDir = "dist\MotionBloom"
if (-not (Test-Path "$appDir\MotionBloom.exe")) {
    throw "PyInstaller did not produce $appDir\MotionBloom.exe"
}

# --- 3. Stage MSIX layout -------------------------------------------------
Write-Host "Staging MSIX layout..."
Copy-Item "packaging\AppxManifest.xml" "$appDir\AppxManifest.xml" -Force

$assetsSrc = "packaging\Assets\store"
$assetsDst = "$appDir\Assets"
New-Item -ItemType Directory -Force -Path $assetsDst | Out-Null
if (Test-Path $assetsSrc) {
    Copy-Item "$assetsSrc\*" $assetsDst -Recurse -Force
}

# Sanity check: manifest references these logo files.
$requiredAssets = @(
    "StoreLogo.png",
    "Square150x150Logo.png",
    "Square44x44Logo.png",
    "Wide310x150Logo.png"
)
foreach ($a in $requiredAssets) {
    if (-not (Test-Path (Join-Path $assetsDst $a))) {
        Write-Warning "Missing asset: $a — MakeAppx will fail. Place it in $assetsSrc."
    }
}

# --- 4. Locate MakeAppx.exe ----------------------------------------------
$sdkRoots = @(
    "C:\Program Files (x86)\Windows Kits\10\bin",
    "C:\Program Files\Windows Kits\10\bin"
)
$makeAppx = $null
foreach ($root in $sdkRoots) {
    if (Test-Path $root) {
        $candidate = Get-ChildItem $root -Recurse -Filter MakeAppx.exe -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match "\\x64\\" } |
            Sort-Object FullName -Descending |
            Select-Object -First 1
        if ($candidate) { $makeAppx = $candidate.FullName; break }
    }
}
if (-not $makeAppx) {
    throw "MakeAppx.exe not found. Install the Windows 10/11 SDK."
}
Write-Host "Using MakeAppx: $makeAppx"

# --- 5. Pack MSIX ---------------------------------------------------------
$msixOut = "dist\MotionBloom.msix"
& $makeAppx pack /d $appDir /p $msixOut /o
if ($LASTEXITCODE -ne 0) { throw "MakeAppx pack failed ($LASTEXITCODE)" }

Write-Host ""
Write-Host "========================================================"
Write-Host " MSIX built: $msixOut"
Write-Host " Upload this file to Partner Center for Product 9PJ9QCB6VWK9."
Write-Host "========================================================"
