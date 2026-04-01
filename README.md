# LocalSTT Portable

Local portable STT application for Windows.

The application runs as a regular 450x600 window and does not use a system tray icon.

## Why This Project

Lightweight and reliable daily dictation tool for Windows.

- Super simple: press hotkey, speak, release, text appears.
- Super fast: local Whisper transcription with no cloud round-trip.
- No installation for end users: download a ZIP from GitHub Releases, extract it, and run.
- Works in any PC app that accepts paste/input.

## What It Does

- Voice recording with a global hotkey
- Recording transcription using OpenAI Whisper models via faster-whisper
- Default stable mode: after recording stops, the whole WAV file is transcribed in one pass
- Experimental mode is still available: near-real-time overlapping chunks during recording for comparison/debugging
- Preferred transcription language can be forced in settings to avoid wrong auto-detection
- Text insertion into the active window
- Recovery actions in the app window: cancel processing, re-paste the last text, and undo the last paste
- Offline mode by default

## Input and Cursor Behavior

- Dictation is inserted at the current caret position in the active window.
- Start and end of one dictation are applied to the same active input target.
- Clipboard can be restored automatically after paste (configurable).

## Performance (Typical)

- With the default `small` model, transcription time is often around half of audio duration on typical modern hardware.
- Example: 20 seconds of audio is often transcribed in about 8-12 seconds.
- Actual speed depends on CPU/GPU, selected model size, and system load.

## Speech Model

- Engine: faster-whisper (CTranslate2 runtime)
- Model family: OpenAI Whisper (multilingual)
- Default runtime model: `models/faster-whisper-small` (bundled in EXE builds)
- Built-in selectable alternative: `models/faster-whisper-medium` (if downloaded)
- Additional supported local model: `models/faster-whisper-tiny` (manual config)

Repositories:
- Whisper (OpenAI): https://github.com/openai/whisper
- faster-whisper (SYSTRAN): https://github.com/SYSTRAN/faster-whisper

Current model options in this project:
- Version 1: `small` (default, faster)
- Version 2: `medium` (higher quality, slower)
- Optional: `tiny` (fastest, lower accuracy; available via `LOCALSTT_MODEL_PATH`)

## Hotkeys

- Ctrl+Shift+A - cancel the current recording and discard audio
- Ctrl+Shift+Q - start/stop recording
- Ctrl+Shift+W - transcribe the last WAV file
- Ctrl+Shift+E - exit

Hotkeys are bound to the physical A / Q / W / E key positions, so they continue to work across keyboard layouts. On a Russian layout these are the same physical keys where Ф / Й / Ц / У are printed.

## Application Window

Size: 450x600

Tabs:
- Microphone
- Description
- Logs (default)
- History

### Microphone Tab

- Select an available microphone
- Refresh and Select buttons
- Microphone test
- Real-time input level meter
- Basic settings:
  - VAD filter
  - Restore clipboard
  - Preferred transcription language

### Description Tab

Brief information about how the app works and which hotkeys are available.

### Logs Tab

Real-time logs in a compact font.

### History Tab

Shows the last 10 transcription results as stacked text blocks in a scrollable view, including timestamp, language, length, and the full transcription text.

## Application Icon

File used:
- src/icon.png

## Install Dependencies (Run from Source)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Run from Source

```powershell
python src/main.py
```

Optional environment variable:

- `LOCALSTT_MODE` - transcription mode: `full-file` (default) or `live-overlap` (experimental)
- `LOCALSTT_CHUNK_SEC` - chunk duration in seconds for the experimental `live-overlap` mode (default: `2.0`)
- `LOCALSTT_CHUNK_OVERLAP_SEC` - overlap between consecutive chunks in the experimental `live-overlap` mode (default: `0.5`)
- `LOCALSTT_LANGUAGE` - preferred Whisper language code, for example `en`, `ru`, `es`, or `auto`

## Portable Build

Build the portable folder:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_portable.ps1
```

Output:
- dist/LocalSTT/LocalSTT.exe

Important:
- Start LocalSTT.exe specifically from the dist/LocalSTT folder
- Do not move only the EXE file by itself; it must stay next to the `_internal` folder inside the full dist/LocalSTT directory

## Recommended Release Package (ZIP)

Build the recommended end-user ZIP package:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_release_zip.ps1
```

Output:
- dist/LocalSTT-Portable.zip

Recommended end-user flow:
1. Download `LocalSTT-Portable.zip` from GitHub Releases.
2. Extract the ZIP to any folder.
3. Open the extracted `LocalSTT` folder.
4. Run `LocalSTT.exe`.

Important:
- Do not move `LocalSTT.exe` out of the extracted `LocalSTT` folder.
- `LocalSTT.exe` requires the `_internal` folder next to it.

Why ZIP is recommended:
- Faster startup than the one-file EXE.
- Fewer antivirus false positives.
- No installation required.

## ZIP + EXE Build

Build both release artifacts in one run:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_release_bundle.ps1
```

Output:
- dist/LocalSTT-Portable.zip
- dist/LocalSTT-OneFile.exe

Use this when you want to publish both the portable ZIP and the one-file EXE together.

The legacy-compatible script also works:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_exe.ps1
```

## Optional One-File EXE Build

Build a single executable file:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_onefile.ps1
```

Output:
- dist/LocalSTT-OneFile.exe

Use this variant only if you specifically want a single downloadable file.

Tradeoffs:
- Simpler to distribute as one file.
- Slower startup because it unpacks at launch.
- More likely to trigger antivirus false positives than the ZIP portable folder.

## Models

Default model used by the app and bundled in builds:
- models/faster-whisper-small

Also supported:
- models/faster-whisper-medium (built-in mapping when `LOCALSTT_MODEL=medium`)
- models/faster-whisper-tiny (manual path via `LOCALSTT_MODEL_PATH`)

## Whisper Language Support

The bundled multilingual Whisper models support 99 languages and can auto-detect the spoken language.

If needed, language can also be forced in code/config (instead of auto-detection).

Supported language codes:

`en, zh, de, es, ru, ko, fr, ja, pt, tr, pl, ca, nl, ar, sv, it, id, hi, fi, vi, he, uk, el, ms, cs, ro, da, hu, ta, no, th, ur, hr, bg, lt, la, mi, ml, cy, sk, te, fa, lv, bn, sr, az, sl, kn, et, mk, br, eu, is, hy, ne, mn, bs, kk, sq, sw, gl, mr, pa, si, km, sn, yo, so, af, oc, ka, be, tg, sd, gu, am, yi, lo, uz, fo, ht, ps, tk, nn, mt, sa, lb, my, bo, tl, mg, as, tt, haw, ln, ha, ba, jw, su`

To pre-download models manually:

```powershell
python scripts/download_models.py
```

## Data Folders

- Recordings: %LOCALAPPDATA%\LocalSTT\recordings
- Logs: %LOCALAPPDATA%\LocalSTT\logs\app.log
- Settings: %LOCALAPPDATA%\LocalSTT\settings.json
