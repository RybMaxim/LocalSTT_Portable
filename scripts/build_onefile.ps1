$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$buildEnvPath = ".build-env"

Write-Host "Cleaning previous one-file build artifacts..."
if (Test-Path "build") {
    Remove-Item -Path "build" -Recurse -Force
}
if (Test-Path "dist\LocalSTT-OneFile.exe") {
    Remove-Item -Path "dist\LocalSTT-OneFile.exe" -Force
}
if (Test-Path "LocalSTT-OneFile.spec") {
    Remove-Item -Path "LocalSTT-OneFile.spec" -Force
}
if (Test-Path $buildEnvPath) {
    Remove-Item -Path $buildEnvPath -Recurse -Force
}

if (-not (Test-Path "models\faster-whisper-small")) {
    throw "Missing required model folder: models\faster-whisper-small"
}

$python = $null
if (Test-Path ".venv\Scripts\python.exe") {
    $python = ".\.venv\Scripts\python.exe"
} else {
    $pythonCommand = Get-Command python -ErrorAction SilentlyContinue
    if ($null -ne $pythonCommand) {
        $python = $pythonCommand.Source
    }
}

if (-not $python) {
    throw "Python executable not found. Activate .venv or add python to PATH."
}

Write-Host "Creating isolated build environment..."
& $python -m venv $buildEnvPath
if ($LASTEXITCODE -ne 0) {
    throw "Failed to create isolated build environment (exit code $LASTEXITCODE)"
}

$python = Join-Path $buildEnvPath "Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "Build environment python not found: $python"
}

& $python -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) {
    throw "Failed to upgrade pip (exit code $LASTEXITCODE)"
}
& $python -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install requirements (exit code $LASTEXITCODE)"
}
& $python -m pip install pyinstaller
if ($LASTEXITCODE -ne 0) {
    throw "Failed to install PyInstaller (exit code $LASTEXITCODE)"
}

# Build one-file window app with bundled default model and icon.
& $python -m PyInstaller `
    --noconfirm `
    --clean `
    --windowed `
    --onefile `
    --name LocalSTT-OneFile `
    --add-data "models\faster-whisper-small;models\faster-whisper-small" `
    --add-data "src\icon.png;src" `
    --collect-data faster_whisper `
    --collect-data assemblyai `
    --collect-submodules assemblyai `
    src/main.py

if ($LASTEXITCODE -ne 0) {
    throw "PyInstaller one-file build failed (exit code $LASTEXITCODE)"
}

Write-Host "One-file build completed: dist\LocalSTT-OneFile.exe"

if (Test-Path $buildEnvPath) {
    Remove-Item -Path $buildEnvPath -Recurse -Force
}
