#!/bin/bash
set -e

# Get project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "Cleaning previous build artifacts..."
rm -rf build dist LocalSTT.spec

if [ ! -d ".venv" ]; then
    echo "Virtual environment not found: .venv. Run Phase 1 initialization first."
    exit 1
fi

source .venv/bin/activate

echo "Ensuring pyinstaller is installed..."
pip install pyinstaller

# Check for model folder
if [ ! -d "models/faster-whisper-small" ]; then
    echo "Warning: models/faster-whisper-small not found. The app might not work out of the box."
    # We could either fail or proceed with warning.
fi

echo "Building macOS application bundle..."

# Build macOS app bundle
pyinstaller \
    --noconfirm \
    --clean \
    --windowed \
    --name "LocalSTT" \
    --osx-bundle-identifier "com.localstt.app" \
    --target-architecture "arm64" \
    --add-data "src/icon.png:src" \
    --collect-data faster_whisper \
    src/main.py

echo "macOS build completed: dist/LocalSTT.app"
