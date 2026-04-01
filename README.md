# LocalSTT Portable

Local portable STT application for Windows and macOS.

The application runs as a regular window and does not use a system tray icon.

## Why This Project

Lightweight and reliable daily dictation tool for Windows and macOS.

- Super simple: press hotkey, speak, release, text appears.
- Super fast: local Whisper transcription with no cloud round-trip.
- Cross-platform: Native support for Windows and macOS (Intel/Apple Silicon).
- No installation for end users: download a ZIP/DMG from GitHub Releases, extract it, and run.
- Works in any app that accepts paste/input.

Please see the [CHANGELOG.md](CHANGELOG.md) for details on the latest updates, architectural changes, and cross-platform developments.

## What It Does

- Voice recording with a global hotkey.
- Recording transcription using OpenAI Whisper models via faster-whisper.
- Default stable mode: after recording stops, the whole WAV file is transcribed in one pass.
- Text insertion into the active window (using native OS APIs).
- Recovery actions in the app window: cancel processing, re-paste the last text, and undo the last paste.
- Offline mode by default.

## Input and Cursor Behavior

- Dictation is inserted at the current caret position in the active window.
- Start and end of one dictation are applied to the same active input target.
- Clipboard is restored automatically after paste.

## Performance (Typical)

- With the default `small` model, transcription time is often around half of audio duration on typical modern hardware.
- On Apple Silicon (M1/M2/M3), transcription is highly optimized using CoreML/Accelerate via faster-whisper.
- Example: 20 seconds of audio is often transcribed in about 5-10 seconds on a Mac.

## Speech Model

- Engine: faster-whisper (CTranslate2 runtime)
- Model family: OpenAI Whisper (multilingual)
- Default runtime model: `models/faster-whisper-small`
- Built-in selectable alternative: `models/faster-whisper-medium` (if downloaded)

## Hotkeys

The application uses different hotkey prefixes depending on the operating system to avoid system conflicts.

| Action | Windows | macOS |
| :--- | :--- | :--- |
| **Start/Stop Recording** | `Ctrl + Alt + Q` | `Ctrl + Option + Q` |
| **Transcribe Last File** | `Ctrl + Alt + W` | `Ctrl + Option + W` |
| **Exit Application** | `Ctrl + Alt + E` | `Ctrl + Option + E` |

## macOS Specific Requirements

### Accessibility Permissions
On macOS, the application requires **Accessibility** permissions to simulate the "Paste" command (`Cmd+V`) and to listen for global hotkeys.

1. When you first run the app and press a hotkey, macOS will prompt you to grant Accessibility permissions.
2. Open **System Settings** > **Privacy & Security** > **Accessibility**.
3. Add or toggle on **LocalSTT** (or your Terminal if running from source).
4. Restart the application for changes to take effect.

### Microphone Access
The app will also request Microphone access upon first recording. Ensure this is granted in **Privacy & Security** > **Microphone**.

## Install Dependencies (Run from Source)

### Windows
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### macOS
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run from Source

```bash
python src/main.py
```

## Portable Build

### Windows
```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_portable.ps1
```

### macOS
```bash
./scripts/build_macos.sh
```
Output: `dist/LocalSTT.app`

## Models

Default model: `models/faster-whisper-small`.

To pre-download models manually:
```bash
python scripts/download_models.py
```

## Data Folders

- **Windows:** `%LOCALAPPDATA%\LocalSTT\`
- **macOS:** `~/Library/Application Support/LocalSTT/`
