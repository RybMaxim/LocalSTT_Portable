# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] - 2026-04-01

### Added
- **Floating Recording HUD:** Implemented a high-visibility, "Always-on-Top" red floating status box that appears during recording. It includes a "● REC" indicator and a real-time timer.
- **Native Dual-Target Architecture:** Introduced `src/os_adapter.py` providing an `OSAdapter` interface to handle platform-specific UI interactions (window focus, paste, undo).
- **macOS Support:** Implemented `MacAdapter` using native `osascript` (AppleScript) for reliable window targeting and simulated keystrokes.
- **macOS Build Script:** Added `scripts/build_macos.sh` to compile the application into a `.app` bundle using PyInstaller.
- **Architecture & Contributing Docs:** Added `ARCHITECTURE.md` and `CONTRIBUTING.md` to establish guidelines for the new cross-platform direction.

### Changed
- **Focus Preservation:** Refactored the notification system to ensure the main application window remains in the background when starting a recording via hotkey. The HUD now uses non-focus-stealing topmost attributes.
- **Hotkey System:** Migrated from a custom key listener to `pynput.keyboard.GlobalHotKeys` for better cross-platform reliability and safety. 
  - Windows bindings remain `Ctrl + Alt + Q/W/E`.
  - macOS bindings map to `Ctrl + Option + Q/W/E` (including Quartz virtual key codes for alternate keyboard layouts).
- **Audio Device Selection:** Improved `_log_audio_input_info` and stream initialization to explicitly verify `max_input_channels > 0`, preventing crashes when output-only devices are selected.
- **Microphone Refresh:** The UI "Refresh" button now completely restarts the PortAudio engine (`sd._terminate()` / `sd._initialize()`) to dynamically detect newly connected Bluetooth devices (e.g., AirPods).

### Fixed
- **macOS Accelerate Bug (NumPy 2.0+):** Applied a startup monkey-patch to `faster_whisper.feature_extractor.FeatureExtractor` converting `mel_filters` to `float32`. This bypasses a known macOS `vecLib` bug with `float16` matrix multiplication that caused severe transcription degradation ("garbage" output) and `RuntimeWarning` overflow logs.
- **Self-Targeting Paste Bug:** Fixed an edge case where executing the hotkey while the LocalSTT application itself was the active window would cause the application to overwrite the clipboard and fail to paste. The app now recognizes itself as the foreground window and safely leaves the transcribed text in the clipboard for manual pasting.

---

## 🛠️ Action Items for Windows Maintainers

As part of the cross-platform pivot, several core systems were refactored. The following systems require regression testing on a native Windows host before merging:

1. **Verify `WindowsAdapter` (`src/os_adapter.py`):** 
   - Ensure `get_foreground_window`, `send_paste`, and `send_undo` still correctly hook into Windows `ctypes.windll.user32` and `pyautogui`.
2. **Verify `GlobalHotKeys` Migration:** 
   - Confirm that `Ctrl + Alt + Q/W/E` binds correctly on Windows using `pynput.keyboard.GlobalHotKeys` and does not conflict with system shortcuts.
3. **Verify Audio Device Enumeration:**
   - Confirm that the new `max_input_channels > 0` validation accurately detects Windows DirectSound, MME, and WASAPI input devices without filtering out valid microphones.
4. **Winget/Scoop Packaging:**
   - As noted in `ARCHITECTURE.md`, begin the transition for Windows packaging pipelines.
