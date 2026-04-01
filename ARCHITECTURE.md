# Architecture: Native Dual-Target Architecture

LocalSTT has transitioned from a Windows-first application to a native dual-target architecture supporting both Windows and macOS.

## Core Design Principles

1.  **OS Abstraction Layer:** All OS-level operations (e.g., window identification, foreground window capture, simulated keystrokes, and clipboard restoration) are abstracted through the `OSAdapter` interface.
2.  **Native Interaction:** Instead of using generic cross-platform libraries for UI automation, LocalSTT utilizes native APIs on each platform (`ctypes` for Windows, `osascript`/AppleScript for macOS) to ensure reliability and performance.
3.  **Core Decoupling:** The transcription logic, audio processing, and Whisper model management remain platform-agnostic.

## Components

### 1. Application Core (`LocalSTTCore`)
Contains the non-UI, platform-agnostic logic for:
-   Audio recording and chunking.
-   Whisper model transcription.
-   Transcription history management.

### 2. OS Adapter (`OSAdapter`)
An abstract base class defined in `src/os_adapter.py`.

-   **`WindowsAdapter`**: Uses `ctypes` to call `User32.dll` and `Kernel32.dll` for window management and keyboard events.
-   **`MacAdapter`**: Uses `osascript` to communicate with `System Events` for window identification and keystroke simulation.

### 3. Application Frontend (`LocalSTTApp`)
Uses `tkinter` for the user interface, which provides a consistent, albeit basic, look and feel across platforms while remaining lightweight.

## Deployment & Packaging

-   **Windows**: Packaged as a portable `.exe` using `pyinstaller`. Future support for Winget and Scoop is planned.
-   **macOS**: Packaged as a `.app` bundle using `pyinstaller`, targeting `arm64` (Apple Silicon). Distribution via Homebrew Cask is the intended path.

## Planned Improvements

-   **Linux Support**: Implementation of a `LinuxAdapter` (potentially using `xdotool` or `ydotool` for Wayland).
-   **Advanced macOS Automation**: Potential transition from `osascript` to `pyobjc` (AppKit) for more granular control if required.
-   **CI/CD Pipeline**: Automated multi-platform builds and testing using GitHub Actions.
