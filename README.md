# LocalSTT

Portable offline speech-to-text application for Windows.

LocalSTT is designed for fast daily dictation: press a hotkey, speak, release the hotkey, and the recognized text is pasted into the active application.

## Overview

- Fully local transcription with `faster-whisper`
- Portable folder-based build, no installer required
- Default bundled Whisper model for immediate use
- Two additional Whisper models can be downloaded from the app UI and used locally
- Global hotkeys for recording, retranscribing the last file, and closing the app
- Recovery actions for cancel, re-paste, and undo
- Real-time logs and transcription history

## Current Distribution Format

The current implementation is centered around the portable folder build.

- Start the app from `dist/LocalSTT/LocalSTT.exe`
- Keep the `_internal` folder next to the EXE
- Do not move the EXE out of the `LocalSTT` folder by itself

The portable build keeps all required runtime files next to the executable, which is also where downloaded speech models are stored.

## Main Features

- Global hotkey recording
- Stable transcription mode: the full recorded WAV file is transcribed after recording stops
- Optional experimental `live-overlap` mode for chunked near-real-time comparison/debugging
- Local multilingual Whisper transcription
- Preferred transcription language setting
- Automatic paste into the active target window
- Optional clipboard restore after paste
- Local recordings, logs, settings, and history

## Hotkeys

- `Ctrl+Shift+A` - cancel the current recording and discard audio
- `Ctrl+Shift+Q` - start/stop recording
- `Ctrl+Shift+W` - transcribe the last recorded WAV file again
- `Ctrl+Shift+E` - exit the application

Russian keyboard layout variants are also supported for the same shortcuts.

Hotkeys are bound to the physical A / Q / W / E key positions, so they continue to work across keyboard layouts. On a Russian layout these are the same physical keys where Ф / Й / Ц / У are printed.

## Application Window

The current UI is a fixed-size desktop window.

- Window size: `500 x 720`
- Tabs:
  - `Microphone`
  - `Description`
  - `Logs`
  - `History`
- The `Logs` tab is selected by default on startup

### Microphone Tab

The `Microphone` tab is scrollable and contains the main working controls.

- Input device dropdown
- Working `Refresh` button for re-scanning microphones connected after startup
- Automatic microphone selection when an item is chosen in the dropdown
- Input level meter
- Microphone test button
- Preferred transcription language selector
- Compact speech-model selector and action button
- `VAD filter` checkbox
- `Restore clipboard` checkbox
- `Save settings` button

Mouse-wheel changes are blocked on the language and model dropdowns so those values do not change accidentally while scrolling.

### Description Tab

Shows a short in-app explanation of the workflow and the hotkeys.

### Logs Tab

Shows runtime logs in real time.

### History Tab

Shows the latest saved transcription results with timestamp, language, mode, and full text.

## Speech Models

LocalSTT uses multilingual Whisper models through `faster-whisper` and the CTranslate2 runtime.

### Model Strategy

The current app supports three model sizes:

| Model | Approx. Size | Availability | Typical Tradeoff |
| --- | ---: | --- | --- |
| `tiny` | 74.6 MB | Download from UI | Fastest, lowest recognition quality |
| `small` | 463.7 MB | Bundled by default | Best default balance of speed and quality |
| `medium` | 1459.7 MB | Download from UI | Better recognition quality, slower and heavier |

### Default Model

`small` is bundled with the portable app.

That means:

- the app can start and transcribe immediately after extraction
- no first-run model download is required for the default setup
- the bundled model is available offline

### Downloadable Models

The two additional models are:

- `tiny`
- `medium`

These models are not required for normal use, but they are available directly from the UI.

#### Why download another model?

Choose `tiny` if you want:

- the smallest disk usage
- the fastest loading and transcription speed
- a lower-quality but lightweight option

Choose `medium` if you want:

- higher recognition quality than `small`
- better handling of more difficult speech
- and you accept higher RAM usage and slower processing

### How Model Download Works in the Current App

The download flow is built into the `Microphone` tab.

1. Open the `Microphone` tab.
2. In the `Speech model` dropdown, choose `Tiny` or `Medium`.
3. If the selected model is not installed yet, the action button changes to `Download Tiny` or `Download Medium`.
4. Click the button.
5. The app downloads the selected model.
6. After the download finishes, the app can activate and use that model locally.

If the selected model is already present, the button changes to a `Use ...` action instead of a download action.

### Where Downloaded Models Are Stored

For the portable build, downloaded models are stored next to the executable inside the runtime folder:

- `dist/LocalSTT/_internal/models`

This keeps the portable build self-contained.

The bundled default model is also resolved from the same runtime area.

### Using a Downloaded Model

After a model is downloaded:

- it becomes available in the same model dropdown
- it can be activated from the same compact control block
- the selected model is saved in the user settings
- future launches reuse the already available local files

### Offline Behavior

The application is offline by default for transcription.

In practice this means:

- transcription itself is local
- cloud speech APIs are not used
- internet is only needed when downloading a model that is not already available locally

### Startup Fallback

If the settings point to a model that is not available anymore, the app falls back to the bundled `small` model so the application can still start.

## Languages

The bundled and downloadable Whisper models are multilingual.

The app supports:

- automatic language detection
- forced language selection from the UI
- manual entry of a Whisper language code

Common examples:

- `en`
- `ru`
- `de`
- `es`
- `fr`
- `pt`
- `uk`
- `ja`
- `zh`

## Recording and Paste Behavior

- Recording starts and stops through the global hotkey
- In the stable mode, a WAV file is created first and transcribed second
- The recognized text is inserted into the active target window
- The same active target is preserved for a single dictation cycle
- Clipboard restore can be enabled or disabled in settings

## Recovery Actions

The main window also includes recovery controls:

- `Cancel processing`
- `Re-paste last text`
- `Undo last paste`

These help when a transcription is still running or when the last paste needs to be repeated or reverted safely.

## Run From Source

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

Start the app:

```powershell
python src/main.py
```

## Build the Portable App

Build the current portable folder version:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/build_portable.ps1
```

Output:

- `dist/LocalSTT/LocalSTT.exe`

Important:

- run the app from inside `dist/LocalSTT`
- keep `_internal` next to the EXE
- downloaded models for the portable build are stored under `_internal/models`

## Optional Environment Variables

These are mainly useful for development and testing.

- `LOCALSTT_MODE` - `full-file` or `live-overlap`
- `LOCALSTT_CHUNK_SEC` - chunk duration for `live-overlap`
- `LOCALSTT_CHUNK_OVERLAP_SEC` - overlap between chunks in `live-overlap`
- `LOCALSTT_LANGUAGE` - preferred Whisper language code
- `LOCALSTT_MODEL` - initial model selection (`tiny`, `small`, `medium`)
- `LOCALSTT_MODEL_PATH` - explicit model path override
- `LOCALSTT_OFFLINE_ONLY` - offline behavior flag
- `LOCALSTT_COMPUTE` - compute type passed to `faster-whisper`
- `LOCALSTT_DEVICE` - device hint passed to `faster-whisper`

## Manual Model Download for Development

If you want to pre-download the model folders in the repository itself for development/build preparation, use:

```powershell
python scripts/download_models.py
```

This is optional and mainly useful when preparing local developer builds.

## Data Locations

LocalSTT stores user data under `%LOCALAPPDATA%\LocalSTT`.

- Recordings: `%LOCALAPPDATA%\LocalSTT\recordings`
- Logs: `%LOCALAPPDATA%\LocalSTT\logs\app.log`
- Settings: `%LOCALAPPDATA%\LocalSTT\settings.json`
- History: `%LOCALAPPDATA%\LocalSTT\history.json`

For the portable build, downloaded runtime models are stored in:

- `dist/LocalSTT/_internal/models`

## Dependencies

Main runtime dependencies:

- `faster-whisper`
- `sounddevice`
- `numpy`
- `pynput`
- `pyperclip`
- `pyautogui`
- `huggingface_hub`

## Summary

The current implementation is a portable local dictation tool with:

- bundled `small` model for immediate offline use
- downloadable `tiny` and `medium` models available directly from the UI
- local model reuse after download
- compact in-window model switching
- no required installer and no cloud transcription dependency