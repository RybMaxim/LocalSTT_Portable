# Contributing to LocalSTT

Thank you for your interest in contributing to LocalSTT Portable! We've recently transitioned to a Native Dual-Target Architecture to better support both Windows and macOS.

## Architectural Overview

The core logic of the application is decoupled from OS-specific operations through an abstraction layer.

### OS Adapter Pattern

All OS-specific interactions (window management, simulated keystrokes, clipboard handling) are defined in `src/os_adapter.py`.

- **OSAdapter (Abstract Base Class):** Defines the required interface for all operating systems.
- **WindowsAdapter:** Implements the interface using `ctypes` for native Windows API calls.
- **MacAdapter:** Implements the interface using `osascript` (AppleScript) for native macOS UI automation.

### Adding Support for New Platforms

To add support for another platform (e.g., Linux), you must:
1. Create a new subclass of `OSAdapter` in `src/os_adapter.py`.
2. Implement all abstract methods using platform-specific APIs.
3. Update the `get_os_adapter` factory function to detect and return your new adapter.

## Development Environment

### Setup

1. Clone the repository.
2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # or .venv\Scripts\activate on Windows
   pip install -r requirements.txt
   ```

### Testing

We use `pytest` for unit testing. All OS-specific logic should have corresponding tests in the `tests/` directory.

To run tests:
```bash
pytest tests
```

### macOS Build

Use the provided build script for macOS:
```bash
./scripts/build_macos.sh
```

## Pull Request Guidelines

1. Ensure all tests pass.
2. Separate Windows and macOS specific changes into logical commits.
3. Follow conventional commit standards.
4. If your PR introduces a new feature, include tests for both supported platforms.
5. Update documentation if necessary.
