import json
import logging
import os
import queue
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyautogui
import sounddevice as sd
from faster_whisper import WhisperModel
from pynput import keyboard
from pynput.keyboard import Controller

from app_core import LocalSTTCore


CTRL_KEYS = {keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r}
SHIFT_KEYS = {keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r}
HOTKEY_SCANS = {
    0x10: "toggle_recording",  # Physical Q key position
    0x11: "transcribe_last",   # Physical W key position
    0x12: "shutdown",          # Physical E key position
}
COMMON_LANGUAGE_OPTIONS: list[tuple[str, str]] = [
    ("Auto detect", "auto"),
    ("English (en)", "en"),
    ("Russian (ru)", "ru"),
    ("Spanish (es)", "es"),
    ("German (de)", "de"),
    ("French (fr)", "fr"),
    ("Italian (it)", "it"),
    ("Portuguese (pt)", "pt"),
    ("Ukrainian (uk)", "uk"),
    ("Polish (pl)", "pl"),
    ("Turkish (tr)", "tr"),
    ("Japanese (ja)", "ja"),
    ("Korean (ko)", "ko"),
    ("Chinese (zh)", "zh"),
]
LANGUAGE_LABEL_TO_CODE = {label: code for label, code in COMMON_LANGUAGE_OPTIONS}
LANGUAGE_CODE_TO_LABEL = {code: label for label, code in COMMON_LANGUAGE_OPTIONS}
COMMON_TRANSCRIPTION_MODE_OPTIONS: list[tuple[str, str]] = [
    ("Full file after stop (stable)", "full-file"),
    ("Live overlap during recording (experimental)", "live-overlap"),
]
TRANSCRIPTION_MODE_LABEL_TO_CODE = {label: code for label, code in COMMON_TRANSCRIPTION_MODE_OPTIONS}
COMMON_MODEL_OPTIONS: list[tuple[str, str]] = [
    ("Tiny (fastest, lowest accuracy)", "tiny"),
    ("Small (balanced, default)", "small"),
    ("Medium (slower, higher accuracy)", "medium"),
]
MODEL_LABEL_TO_CODE = {label: code for label, code in COMMON_MODEL_OPTIONS}
MODEL_CODE_TO_LABEL = {code: label for label, code in COMMON_MODEL_OPTIONS}
MODEL_CODE_TO_PATH = {
    "tiny": "models/faster-whisper-tiny",
    "small": "models/faster-whisper-small",
    "medium": "models/faster-whisper-medium",
}
TRANSCRIPTION_HISTORY_LIMIT = 10
SUPPORTED_LANGUAGE_CODES = {
    "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr", "pl", "ca", "nl", "ar", "sv",
    "it", "id", "hi", "fi", "vi", "he", "uk", "el", "ms", "cs", "ro", "da", "hu", "ta", "no",
    "th", "ur", "hr", "bg", "lt", "la", "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr",
    "az", "sl", "kn", "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk", "sq", "sw",
    "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc", "ka", "be", "tg", "sd", "gu",
    "am", "yi", "lo", "uz", "fo", "ht", "ps", "tk", "nn", "mt", "sa", "lb", "my", "bo", "tl",
    "mg", "as", "tt", "haw", "ln", "ha", "ba", "jw", "su",
}


def _normalize_transcription_language(value: Any, fallback: str = "auto") -> str:
    if value is None:
        return fallback

    raw = str(value).strip()
    if not raw:
        return fallback
    if raw in LANGUAGE_LABEL_TO_CODE:
        return LANGUAGE_LABEL_TO_CODE[raw]

    cleaned = raw.lower()
    if cleaned in {"auto", "auto detect", "detect"}:
        return "auto"
    if cleaned in SUPPORTED_LANGUAGE_CODES:
        return cleaned
    return fallback


def _language_display_value(language_code: str) -> str:
    normalized = _normalize_transcription_language(language_code)
    return LANGUAGE_CODE_TO_LABEL.get(normalized, normalized)


def _normalize_transcription_mode(value: Any, fallback: str = "full-file") -> str:
    if value is None:
        return fallback

    raw = str(value).strip()
    if not raw:
        return fallback
    if raw in TRANSCRIPTION_MODE_LABEL_TO_CODE:
        return TRANSCRIPTION_MODE_LABEL_TO_CODE[raw]

    cleaned = raw.lower()
    if cleaned in {"full", "full-file", "full_file", "stable", "single", "single-file"}:
        return "full-file"
    if cleaned in {"live", "live-overlap", "live_overlap", "stream", "semi-stream", "semi-streaming"}:
        return "live-overlap"
    return fallback


def _normalize_model_size(value: Any, fallback: str = "small") -> str:
    if value is None:
        return fallback

    raw = str(value).strip()
    if not raw:
        return fallback
    if raw in MODEL_LABEL_TO_CODE:
        return MODEL_LABEL_TO_CODE[raw]

    cleaned = raw.lower().replace("_", "-")
    if cleaned in MODEL_CODE_TO_LABEL:
        return cleaned
    if cleaned.endswith("faster-whisper-tiny") or cleaned.endswith("/tiny") or cleaned.endswith("\\tiny"):
        return "tiny"
    if cleaned.endswith("faster-whisper-small") or cleaned.endswith("/small") or cleaned.endswith("\\small"):
        return "small"
    if cleaned.endswith("faster-whisper-medium") or cleaned.endswith("/medium") or cleaned.endswith("\\medium"):
        return "medium"
    return fallback


def _model_size_from_path(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    raw = str(value).strip()
    if not raw:
        return fallback
    normalized = raw.replace("\\", "/").rstrip("/").lower()
    if normalized.endswith("faster-whisper-tiny"):
        return "tiny"
    if normalized.endswith("faster-whisper-small"):
        return "small"
    if normalized.endswith("faster-whisper-medium"):
        return "medium"
    return fallback


def _model_display_value(model_size: str) -> str:
    normalized = _normalize_model_size(model_size)
    return MODEL_CODE_TO_LABEL.get(normalized, normalized)


def _model_path_for_size(model_size: str) -> str:
    normalized = _normalize_model_size(model_size)
    return MODEL_CODE_TO_PATH.get(normalized, MODEL_CODE_TO_PATH["small"])


@dataclass
class AppConfig:
    sample_rate: int = 16000
    channels: int = 1
    dtype: str = "int16"
    model_size: str = "small"
    model_path: str = "models/faster-whisper-small"
    offline_only: bool = True
    compute_type: str = "int8"
    device: str = "auto"
    beam_size: int = 5
    vad_filter: bool = True
    paste_delay_sec: float = 0.2
    restore_clipboard: bool = True
    input_device: str | None = None
    popup_duration_sec: float = 1.2
    transcription_mode: str = "full-file"
    chunk_duration_sec: float = 2.0
    chunk_overlap_sec: float = 0.5
    transcription_language: str = "auto"


class QueueLogHandler(logging.Handler):
    def __init__(self, q: queue.Queue[str]) -> None:
        super().__init__()
        self.q = q

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.q.put(self.format(record))
        except Exception:
            pass


class LocalSTTApp(LocalSTTCore):
    def __init__(self, config: AppConfig) -> None:
        self.config = config

        if getattr(sys, "frozen", False):
            self.project_root = Path(sys.executable).resolve().parent
        else:
            self.project_root = Path(__file__).resolve().parent.parent

        local_app_data = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "LocalSTT"
        self.recordings_dir = local_app_data / "recordings"
        self.logs_dir = local_app_data / "logs"
        self.settings_file = local_app_data / "settings.json"
        self.history_file = local_app_data / "history.json"
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        self.log_queue: queue.Queue[str] = queue.Queue()
        self._configure_logging(self.logs_dir / "app.log")

        pyautogui.FAILSAFE = False

        self.is_recording = False
        self.is_transcribing = False
        self.stop_event = threading.Event()

        self.audio_chunks: list[np.ndarray] = []
        self.audio_queue: queue.Queue[np.ndarray] = queue.Queue()
        self.popup_queue: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()
        self.stream: sd.InputStream | None = None
        self.current_recording_sample_rate: int = self.config.sample_rate
        self.mic_monitor_stream: sd.InputStream | None = None
        self.mic_level: float = 0.0

        self.last_audio_file: Path | None = None
        self.keyboard_controller = Controller()
        self.target_hwnd: int | None = None
        self.target_focus_hwnd: int | None = None
        self.action_lock = threading.Lock()
        self.recording_started_at: float | None = None
        self.transcription_started_at: float | None = None
        self.transcription_target_duration_sec: float | None = None
        self.transcription_cancel_event = threading.Event()
        self.last_pasted_text: str | None = None
        self.last_paste_target_hwnd: int | None = None
        self.last_paste_target_focus_hwnd: int | None = None
        self.last_paste_can_undo = False
        self.hotkey_ctrl_pressed = False
        self.hotkey_shift_pressed = False
        self.active_hotkey_names: set[str] = set()
        self.recording_audio_lock = threading.Lock()
        self.recording_audio_event = threading.Event()
        self.recording_audio_bytes = bytearray()
        self.recording_audio_frame_count = 0
        self.live_transcription_thread = None
        self.live_transcription_text = ""
        self.live_transcription_chunk_count = 0
        self.live_transcription_language_scores: dict[str, float] = {}
        self.live_transcription_error = None
        self.live_transcription_cancelled = False
        self.live_mode_session = None
        self.transcription_history: list[dict[str, Any]] = []

        self.ui_root = None
        self.ui_log_text = None
        self.ui_tabs = None
        self.ui_mic_var = None
        self.ui_mic_combo = None
        self.ui_mic_level_var = None
        self.ui_mic_test_btn = None
        self.ui_vad_var = None
        self.ui_restore_clipboard_var = None
        self.ui_model_var = None
        self.ui_model_combo = None
        self.ui_language_var = None
        self.ui_language_combo = None
        self.ui_status_var = None
        self.ui_toast_window = None
        self.ui_toast_label = None
        self.ui_toast_timer_id = None
        self.ui_popup_timer_mode: str | None = None
        self.ui_popup_base_text = ""
        self.ui_cancel_transcription_btn = None
        self.ui_repeat_paste_btn = None
        self.ui_undo_paste_btn = None
        self.ui_history_text = None

        self._load_settings_from_file()
        self._sync_model_config()
        self._load_history_from_file()

        self.model = self._load_whisper_model()

        self.input_device = self._resolve_input_device(self.config.input_device)
        self._log_audio_input_info()

        self.hotkey_actions_by_vk: dict[int, tuple[str, Any]] = {
            0x51: ("toggle_recording", self.toggle_recording),
            0x57: ("transcribe_last", self.transcribe_last_file),
            0x45: ("shutdown", self.shutdown),
        }
        self.hotkey_actions_by_scan: dict[int, tuple[str, Any]] = {
            0x10: ("toggle_recording", self.toggle_recording),
            0x11: ("transcribe_last", self.transcribe_last_file),
            0x12: ("shutdown", self.shutdown),
        }

        self.hotkeys = keyboard.Listener(
            on_press=self._on_hotkey_press,
            on_release=self._on_hotkey_release,
        )

    def _key_scan(self, key) -> int | None:
        try:
            scan = getattr(key, "_scan", None)
            if scan is None:
                value = getattr(key, "value", None)
                scan = getattr(value, "_scan", None)
            if scan is None:
                return None
            return int(scan)
        except Exception:
            return None

    def _key_vk(self, key) -> int | None:
        try:
            vk = getattr(key, "vk", None)
            if vk is None:
                return None
            return int(vk)
        except Exception:
            return None

    def _resolve_hotkey_action(self, key) -> tuple[str, Any] | None:
        scan = self._key_scan(key)
        if scan is not None and scan in self.hotkey_actions_by_scan:
            return self.hotkey_actions_by_scan[scan]

        vk = self._key_vk(key)
        if vk is not None and vk in self.hotkey_actions_by_vk:
            return self.hotkey_actions_by_vk[vk]

        return None

    def _on_hotkey_press(self, key) -> None:
        try:
            if key in CTRL_KEYS:
                self.hotkey_ctrl_pressed = True
                return
            if key in SHIFT_KEYS:
                self.hotkey_shift_pressed = True
                return

            if not (self.hotkey_ctrl_pressed and self.hotkey_shift_pressed):
                return

            resolved = self._resolve_hotkey_action(key)
            if resolved is None:
                return

            hotkey_name, action = resolved
            if hotkey_name in self.active_hotkey_names:
                return

            self.active_hotkey_names.add(hotkey_name)
            self._dispatch_hotkey(hotkey_name, action)
        except Exception:
            logging.exception("Hotkey press handler failed")

    def _on_hotkey_release(self, key) -> None:
        try:
            if key in CTRL_KEYS:
                self.hotkey_ctrl_pressed = False
                self.active_hotkey_names.clear()
                return
            if key in SHIFT_KEYS:
                self.hotkey_shift_pressed = False
                self.active_hotkey_names.clear()
                return

            resolved = self._resolve_hotkey_action(key)
            if resolved is None:
                return

            hotkey_name, _action = resolved
            self.active_hotkey_names.discard(hotkey_name)
        except Exception:
            logging.exception("Hotkey release handler failed")

    def _resolve_icon_path(self) -> Path | None:
        candidates = [
            Path(__file__).resolve().parent / "icon.png",
            self.project_root / "src" / "icon.png",
        ]
        for root in self._runtime_roots():
            candidates.append(root / "src" / "icon.png")

        for p in candidates:
            if p.exists():
                return p
        return None

    def _configure_logging(self, log_file: Path) -> None:
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

        queue_handler = QueueLogHandler(self.log_queue)
        queue_handler.setFormatter(formatter)
        logger.addHandler(queue_handler)

    def _sync_model_config(self) -> None:
        fallback_model_size = _model_size_from_path(self.config.model_path, fallback="small")
        self.config.model_size = _normalize_model_size(self.config.model_size, fallback=fallback_model_size)
        if _model_size_from_path(self.config.model_path, fallback=""):
            self.config.model_path = _model_path_for_size(self.config.model_size)

    def _available_model_codes(self) -> list[str]:
        available_codes: list[str] = []
        for code, model_path in MODEL_CODE_TO_PATH.items():
            if self._resolve_existing_path(model_path) is not None:
                available_codes.append(code)

        current_code = _normalize_model_size(
            self.config.model_size,
            fallback=_model_size_from_path(self.config.model_path, fallback=""),
        )
        if current_code and current_code not in available_codes:
            available_codes.insert(0, current_code)

        return available_codes or list(MODEL_CODE_TO_PATH)

    def _set_model_ui_value(self, model_size: str) -> None:
        if self.ui_root is None or self.ui_model_var is None:
            return
        label = _model_display_value(model_size)
        try:
            self.ui_root.after(0, lambda: self.ui_model_var.set(label))
        except Exception:
            pass

    def _apply_selected_model(self, model_size: str) -> None:
        normalized = _normalize_model_size(model_size, fallback="")
        if not normalized:
            raise ValueError("Unsupported model selection")
        self.config.model_size = normalized
        self.config.model_path = _model_path_for_size(normalized)

    def _load_whisper_model(self) -> WhisperModel:
        model_source, local_files_only = self._resolve_model_source()
        if self.config.offline_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

        logging.info("Loading faster-whisper model from: %s", model_source)
        model = WhisperModel(
            model_source,
            device=self.config.device,
            compute_type=self.config.compute_type,
            local_files_only=local_files_only,
        )
        logging.info("Model loaded")
        return model

    def _apply_user_settings(
        self,
        *,
        vad_filter: bool,
        restore_clipboard: bool,
        transcription_language: str,
        model_size: str,
    ) -> None:
        previous_model_size = self.config.model_size
        previous_model_path = self.config.model_path

        self.config.vad_filter = vad_filter
        self.config.restore_clipboard = restore_clipboard
        self.config.transcription_language = transcription_language

        normalized_model_size = _normalize_model_size(model_size, fallback="")
        if not normalized_model_size:
            self._set_status("Unsupported model")
            self._show_popup("UNSUPPORTED MODEL", bg="#9a1b1b")
            self._set_model_ui_value(previous_model_size)
            return

        model_changed = normalized_model_size != previous_model_size
        if not model_changed:
            self._save_settings_to_file()
            self._set_status("Settings saved")
            return

        if self.is_recording or self.is_transcribing:
            self._save_settings_to_file()
            self._set_model_ui_value(previous_model_size)
            self._set_status("Stop the active job before switching model")
            self._show_popup("STOP ACTIVE JOB FIRST", bg="#9a1b1b")
            return

        self._set_status(f"Loading {normalized_model_size} model...")
        self._show_popup(f"LOADING {normalized_model_size.upper()} MODEL", bg="#5d2f87", persistent=True)

        try:
            self._apply_selected_model(normalized_model_size)
            new_model = self._load_whisper_model()
        except Exception:
            self.config.model_size = previous_model_size
            self.config.model_path = previous_model_path
            self._save_settings_to_file()
            self._set_model_ui_value(previous_model_size)
            self._hide_popup()
            logging.exception("Failed to switch model")
            self._set_status(f"Failed to load {normalized_model_size} model")
            self._show_popup("MODEL LOAD FAILED", bg="#9a1b1b")
            return

        self.model = new_model
        self._save_settings_to_file()
        self._set_model_ui_value(self.config.model_size)
        self._hide_popup()
        self._set_status(f"Model switched to {self.config.model_size}")
        self._show_popup(f"MODEL: {self.config.model_size.upper()}", bg="#1e6b2d")

    def _load_settings_from_file(self) -> None:
        try:
            if not self.settings_file.exists():
                return
            with self.settings_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            self.config.input_device = data.get("input_device", self.config.input_device)
            self.config.vad_filter = bool(data.get("vad_filter", self.config.vad_filter))
            self.config.restore_clipboard = bool(data.get("restore_clipboard", self.config.restore_clipboard))
            self.config.model_size = str(data.get("model_size", self.config.model_size))
            self.config.model_path = str(data.get("model_path", self.config.model_path))
            self.config.transcription_mode = _normalize_transcription_mode(
                data.get("transcription_mode", self.config.transcription_mode),
                fallback=self.config.transcription_mode,
            )
            self.config.transcription_language = _normalize_transcription_language(
                data.get("transcription_language", self.config.transcription_language),
                fallback=self.config.transcription_language,
            )
        except Exception:
            logging.exception("Failed to load settings")

    def _save_settings_to_file(self) -> None:
        try:
            payload: dict[str, Any] = {
                "input_device": self.config.input_device,
                "vad_filter": self.config.vad_filter,
                "restore_clipboard": self.config.restore_clipboard,
                "model_size": self.config.model_size,
                "model_path": self.config.model_path,
                "transcription_mode": self.config.transcription_mode,
                "transcription_language": self.config.transcription_language,
            }
            with self.settings_file.open("w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
        except Exception:
            logging.exception("Failed to save settings")

    def _load_history_from_file(self) -> None:
        try:
            if not self.history_file.exists():
                self.transcription_history = []
                return
            with self.history_file.open("r", encoding="utf-8") as f:
                raw_history = json.load(f)

            if not isinstance(raw_history, list):
                self.transcription_history = []
                return

            history: list[dict[str, Any]] = []
            for raw_entry in raw_history[:TRANSCRIPTION_HISTORY_LIMIT]:
                if not isinstance(raw_entry, dict):
                    continue
                text = str(raw_entry.get("text", "")).strip()
                if not text:
                    continue
                history.append(
                    {
                        "created_at": str(raw_entry.get("created_at", "")),
                        "text": text,
                        "char_count": int(raw_entry.get("char_count", len(text)) or len(text)),
                        "language": str(raw_entry.get("language", "unknown") or "unknown"),
                        "mode": str(raw_entry.get("mode", "full-file") or "full-file"),
                    }
                )
            self.transcription_history = history
        except Exception:
            logging.exception("Failed to load transcription history")
            self.transcription_history = []

    def _save_history_to_file(self) -> None:
        try:
            with self.history_file.open("w", encoding="utf-8") as f:
                json.dump(self.transcription_history[:TRANSCRIPTION_HISTORY_LIMIT], f, ensure_ascii=False, indent=2)
        except Exception:
            logging.exception("Failed to save transcription history")

    def _remember_transcription_result(self, *, text: str, language: str, mode: str) -> None:
        cleaned_text = str(text).strip()
        if not cleaned_text:
            return

        entry = {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "text": cleaned_text,
            "char_count": len(cleaned_text),
            "language": str(language or "unknown"),
            "mode": str(mode or "full-file"),
        }
        self.transcription_history = [entry] + self.transcription_history[: TRANSCRIPTION_HISTORY_LIMIT - 1]
        self._save_history_to_file()
        if self.ui_root is not None:
            try:
                self.ui_root.after(0, self._refresh_history_view)
            except Exception:
                logging.exception("Failed to schedule history refresh")

    def _refresh_history_view(self) -> None:
        if self.ui_history_text is None:
            return
        self.ui_history_text.configure(state="normal")
        self.ui_history_text.delete("1.0", "end")

        if not self.transcription_history:
            self.ui_history_text.insert("1.0", "No recent transcriptions yet.")
            self.ui_history_text.configure(state="disabled")
            return

        history_count = len(self.transcription_history)
        for idx, entry in enumerate(self.transcription_history, start=1):
            header = (
                f"#{idx}  {entry.get('created_at', '')}\n"
                f"Chars: {entry.get('char_count', 0)} | "
                f"Language: {entry.get('language', 'unknown')} | "
                f"Mode: {entry.get('mode', 'full-file')}\n\n"
            )
            self.ui_history_text.insert("end", header)
            self.ui_history_text.insert("end", entry.get("text", ""))
            if idx < history_count:
                self.ui_history_text.insert("end", "\n" + ("-" * 28) + "\n\n")
            else:
                self.ui_history_text.insert("end", "\n")

        self.ui_history_text.configure(state="disabled")

    def _dispatch_hotkey(self, hotkey_name: str, action) -> None:
        logging.info("Hotkey pressed: %s", hotkey_name)

        def _runner() -> None:
            try:
                with self.action_lock:
                    action()
            except Exception:
                logging.exception("Hotkey action failed: %s", hotkey_name)

        threading.Thread(target=_runner, daemon=True).start()

    def _dispatch_ui_action(self, action_name: str, action) -> None:
        logging.info("UI action triggered: %s", action_name)

        def _runner() -> None:
            try:
                with self.action_lock:
                    action()
            except Exception:
                logging.exception("UI action failed: %s", action_name)

        threading.Thread(target=_runner, daemon=True).start()

    def _update_recovery_buttons(self) -> None:
        allow_recovery_actions = (not self.is_transcribing) and (not self.is_recording)

        if self.ui_cancel_transcription_btn is not None:
            if self.is_transcribing:
                self.ui_cancel_transcription_btn.state(["!disabled"])
            else:
                self.ui_cancel_transcription_btn.state(["disabled"])

        if self.ui_repeat_paste_btn is not None:
            if self.last_pasted_text and allow_recovery_actions:
                self.ui_repeat_paste_btn.state(["!disabled"])
            else:
                self.ui_repeat_paste_btn.state(["disabled"])

        if self.ui_undo_paste_btn is not None:
            if self.last_paste_can_undo and allow_recovery_actions:
                self.ui_undo_paste_btn.state(["!disabled"])
            else:
                self.ui_undo_paste_btn.state(["disabled"])

    def _set_status(self, text: str) -> None:
        if self.ui_status_var is not None:
            try:
                self.ui_status_var.set(text)
            except Exception:
                pass
        logging.info("STATUS: %s", text)

    def _show_popup(
        self,
        text: str,
        bg: str = "#1e6b2d",
        persistent: bool = False,
        timer_mode: str | None = None,
    ) -> None:
        self.popup_queue.put(
            ("show", {"text": text, "bg": bg, "persistent": persistent, "timer_mode": timer_mode})
        )

    def _hide_popup(self) -> None:
        self.popup_queue.put(("hide", {}))

    def _process_popup_queue(self) -> None:
        if self.ui_toast_window is None or self.ui_toast_label is None or self.ui_root is None:
            return

        def _cancel_timer() -> None:
            if self.ui_toast_timer_id is not None:
                try:
                    self.ui_root.after_cancel(self.ui_toast_timer_id)
                except Exception:
                    pass
                self.ui_toast_timer_id = None

        while True:
            try:
                action, payload = self.popup_queue.get_nowait()
            except queue.Empty:
                break

            if action == "show":
                _cancel_timer()
                text = str(payload.get("text", ""))
                bg = str(payload.get("bg", "#1e6b2d"))
                persistent = bool(payload.get("persistent", False))
                timer_mode = payload.get("timer_mode")
                if timer_mode is None:
                    self.ui_popup_timer_mode = None
                else:
                    normalized_timer_mode = str(timer_mode).strip().lower()
                    self.ui_popup_timer_mode = normalized_timer_mode or None
                self.ui_popup_base_text = text

                self.ui_toast_label.configure(text=text, bg=bg)
                self.ui_toast_window.configure(bg=bg)
                self.ui_toast_window.deiconify()
                self.ui_toast_window.lift()

                if not persistent:
                    timeout_ms = int(self.config.popup_duration_sec * 1000)

                    def _hide_after() -> None:
                        if self.ui_toast_window is not None:
                            self.ui_toast_window.withdraw()
                        self.ui_toast_timer_id = None

                    self.ui_toast_timer_id = self.ui_root.after(timeout_ms, _hide_after)

            elif action == "hide":
                _cancel_timer()
                self.ui_popup_timer_mode = None
                self.ui_popup_base_text = ""
                self.ui_toast_window.withdraw()

    def _format_popup_elapsed(self, elapsed_sec: int, target_sec: float | None = None) -> str:
        minutes = elapsed_sec // 60
        seconds = elapsed_sec % 60
        if target_sec is None or target_sec <= 0:
            return f"{minutes:02d}:{seconds:02d}"

        target_total_sec = max(0, int(round(target_sec)))
        target_minutes = target_total_sec // 60
        target_seconds = target_total_sec % 60
        return f"{minutes:02d}:{seconds:02d} / {target_minutes:02d}:{target_seconds:02d}"

    def _update_recording_popup_timer(self) -> None:
        if self.ui_toast_window is None or self.ui_toast_label is None:
            return

        if self.ui_popup_timer_mode == "recording":
            if not self.is_recording or self.recording_started_at is None:
                return
            elapsed_sec = max(0, int(time.perf_counter() - self.recording_started_at))
            timer_text = self._format_popup_elapsed(elapsed_sec)
            base_text = self.ui_popup_base_text or "RECORDING..."
            self.ui_toast_label.configure(text=f"{base_text} {timer_text}")
            return

        if self.ui_popup_timer_mode == "transcribing":
            if not self.is_transcribing or self.transcription_started_at is None:
                return
            elapsed_sec = max(0, int(time.perf_counter() - self.transcription_started_at))
            timer_text = self._format_popup_elapsed(elapsed_sec, self.transcription_target_duration_sec)
            base_text = self.ui_popup_base_text or "TRANSCRIBING..."
            self.ui_toast_label.configure(text=f"{base_text} {timer_text}")

    def _mic_test_callback(self, indata: np.ndarray, frames: int, callback_time, status) -> None:
        if status:
            logging.warning("Mic test status: %s", status)
        try:
            data = indata.astype(np.float32)
            if data.size == 0:
                self.mic_level = 0.0
                return

            sample_dtype = np.dtype(self.config.dtype)
            if np.issubdtype(sample_dtype, np.integer):
                info = np.iinfo(sample_dtype)
                scale = float(max(abs(info.min), info.max))
            else:
                scale = 1.0

            if scale <= 0:
                self.mic_level = 0.0
                return

            normalized = np.clip(data / scale, -1.0, 1.0)
            rms = float(np.sqrt(np.mean(np.square(normalized))))
            peak = float(np.max(np.abs(normalized)))

            floor_db = -55.0
            rms_db = 20.0 * float(np.log10(max(rms, 1e-6)))
            peak_db = 20.0 * float(np.log10(max(peak, 1e-6)))

            rms_level = ((rms_db - floor_db) / abs(floor_db)) * 100.0
            peak_level = ((peak_db - floor_db) / abs(floor_db)) * 100.0
            instant_level = max(rms_level * 0.65, peak_level)
            instant_level = min(100.0, max(0.0, instant_level))

            if instant_level >= self.mic_level:
                self.mic_level = (self.mic_level * 0.35) + (instant_level * 0.65)
            else:
                self.mic_level = (self.mic_level * 0.82) + (instant_level * 0.18)
        except Exception:
            self.mic_level = 0.0

    def _start_mic_test(self) -> None:
        if self.mic_monitor_stream is not None:
            return
        try:
            device = self._get_effective_input_device()
            if device is None:
                self._set_status("No microphone available")
                return
            self.mic_monitor_stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype=self.config.dtype,
                device=device,
                callback=self._mic_test_callback,
            )
            self.mic_monitor_stream.start()
            self._set_status("Microphone test enabled")
        except Exception:
            logging.exception("Failed to start mic test")
            self._set_status("Failed to start microphone test")
            self.mic_monitor_stream = None

    def _stop_mic_test(self) -> None:
        if self.mic_monitor_stream is None:
            return
        try:
            self.mic_monitor_stream.stop()
            self.mic_monitor_stream.close()
        except Exception:
            pass
        self.mic_monitor_stream = None
        self.mic_level = 0.0
        self._set_status("Microphone test disabled")

    def _toggle_mic_test(self) -> None:
        if self.mic_monitor_stream is None:
            self._start_mic_test()
            if self.ui_mic_test_btn is not None:
                self.ui_mic_test_btn.configure(text="Stop test")
        else:
            self._stop_mic_test()
            if self.ui_mic_test_btn is not None:
                self.ui_mic_test_btn.configure(text="Test microphone")

    def _apply_selected_mic(self) -> None:
        if self.ui_mic_var is None:
            return
        value = self.ui_mic_var.get().strip()
        if not value:
            return
        idx = value.split(":", 1)[0].strip()
        self.config.input_device = idx
        self.input_device = self._resolve_input_device(idx)
        self._save_settings_to_file()
        self._set_status(f"Microphone selected: {idx}")
        self._log_audio_input_info()

    def _refresh_mic_devices(self) -> None:
        if self.ui_mic_combo is None:
            return
        items = [f"{idx}: {name}" for idx, name in self._input_devices_list()]
        self.ui_mic_combo["values"] = items

        selected = str(self.config.input_device) if self.config.input_device is not None else ""
        if selected:
            for item in items:
                if item.startswith(f"{selected}:"):
                    self.ui_mic_var.set(item)
                    break
        elif items:
            self.ui_mic_var.set(items[0])

    def _copy_text_widget_selection(self, widget) -> str:
        try:
            selected_text = widget.get("sel.first", "sel.last")
        except Exception:
            return "break"

        try:
            widget.clipboard_clear()
            widget.clipboard_append(selected_text)
        except Exception:
            logging.exception("Failed to copy selected text from UI widget")
        return "break"

    def _handle_readonly_text_copy_shortcut(self, widget, event) -> str | None:
        keycode = getattr(event, "keycode", None)
        keysym = str(getattr(event, "keysym", "") or "").casefold()
        char = str(getattr(event, "char", "") or "").casefold()

        if keycode == 67:
            return self._copy_text_widget_selection(widget)
        if keysym in {"c", "с", "cyrillic_es"}:
            return self._copy_text_widget_selection(widget)
        if char in {"c", "с"}:
            return self._copy_text_widget_selection(widget)
        return None

    def _bind_readonly_text_widget(self, widget) -> None:
        widget.configure(exportselection=False, takefocus=True)

        def _focus_widget(_event) -> None:
            try:
                widget.focus_set()
            except Exception:
                pass

        widget.bind("<ButtonRelease-1>", _focus_widget, add="+")
        widget.bind(
            "<Control-KeyPress>",
            lambda event: self._handle_readonly_text_copy_shortcut(widget, event),
            add="+",
        )
        widget.bind("<Control-Insert>", lambda _event: self._copy_text_widget_selection(widget), add="+")

    def _ui_poll(self) -> None:
        if self.stop_event.is_set() or self.ui_root is None:
            return

        while True:
            try:
                line = self.log_queue.get_nowait()
            except queue.Empty:
                break
            if self.ui_log_text is not None:
                self.ui_log_text.configure(state="normal")
                self.ui_log_text.insert("end", line + "\n")
                self.ui_log_text.see("end")
                self.ui_log_text.configure(state="disabled")

        if self.ui_mic_level_var is not None:
            self.ui_mic_level_var.set(self.mic_level)

        self._process_popup_queue()
        self._update_recording_popup_timer()
        self._update_recovery_buttons()

        self.ui_root.after(100, self._ui_poll)

    def _configure_ui_styles(self, root) -> None:
        from tkinter import ttk

        style = ttk.Style(root)
        default_bg = style.lookup("TFrame", "background") or root.cget("bg")

        style.configure("AppShell.TFrame", background=default_bg)
        style.configure(
            "App.TNotebook",
            background=default_bg,
            borderwidth=1,
            tabmargins=(2, 2, 2, 0),
        )
        style.configure(
            "App.TNotebook.Tab",
            padding=(10, 4),
            borderwidth=1,
            relief="flat",
            font=("Segoe UI", 9),
        )
        style.map(
            "App.TNotebook.Tab",
            background=[
                ("selected", "#ffffff"),
                ("active", "#f5f5f5"),
            ],
            foreground=[
                ("selected", "#1f1f1f"),
                ("active", "#1f1f1f"),
            ],
            expand=[("selected", [0, 0, 0, 0])],
        )
        style.configure(
            "TabBody.TFrame",
            background="#ffffff",
            borderwidth=1,
            relief="solid",
        )

    def _build_ui(self) -> None:
        import tkinter as tk
        from tkinter import ttk

        root = tk.Tk()
        self.ui_root = root
        root.title("LocalSTT")

        width = 500
        height = 720
        screen_w = root.winfo_screenwidth()
        screen_h = root.winfo_screenheight()
        margin_right = 24

        x = max(0, screen_w - width - margin_right)
        y = max(0, (screen_h - height) // 2)

        root.geometry(f"{width}x{height}+{x}+{y}")
        root.minsize(width, height)
        root.maxsize(width, height)
        self._configure_ui_styles(root)

        icon_path = self._resolve_icon_path()
        if icon_path is not None:
            try:
                icon_img = tk.PhotoImage(file=str(icon_path))
                root.iconphoto(True, icon_img)
                root._icon_img_ref = icon_img
            except Exception:
                logging.warning("Failed to set window icon")

        # Top floating notification line.
        toast = tk.Toplevel(root)
        toast.overrideredirect(True)
        toast.attributes("-topmost", True)
        toast.withdraw()

        toast_w = 460
        toast_h = 46
        toast_x = max(0, (screen_w - toast_w) // 2)
        toast_y = 40
        toast.geometry(f"{toast_w}x{toast_h}+{toast_x}+{toast_y}")

        toast_label = tk.Label(
            toast,
            text="",
            bg="#1e6b2d",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            padx=10,
            pady=8,
        )
        toast_label.pack(fill="both", expand=True)

        self.ui_toast_window = toast
        self.ui_toast_label = toast_label

        container = ttk.Frame(root, padding=8, style="AppShell.TFrame")
        container.pack(fill="both", expand=True)

        notebook = ttk.Notebook(container, style="App.TNotebook")
        notebook.pack(fill="both", expand=True, pady=(2, 0))
        self.ui_tabs = notebook

        tab_mic = ttk.Frame(notebook, style="TabBody.TFrame")
        tab_desc = ttk.Frame(notebook, style="TabBody.TFrame")
        tab_logs = ttk.Frame(notebook, style="TabBody.TFrame")
        tab_history = ttk.Frame(notebook, style="TabBody.TFrame")
        notebook.add(tab_mic, text="Microphone")
        notebook.add(tab_desc, text="Description")
        notebook.add(tab_logs, text="Logs")
        notebook.add(tab_history, text="History")

        log_text = tk.Text(tab_logs, wrap="word", font=("Consolas", 8), state="disabled")
        log_text.pack(fill="both", expand=True, padx=6, pady=6)
        self._bind_readonly_text_widget(log_text)
        self.ui_log_text = log_text

        history_frame = ttk.Frame(tab_history, padding=6)
        history_frame.pack(fill="both", expand=True)

        ttk.Label(history_frame, text="Last 10 transcriptions").pack(anchor="w", pady=(0, 6))

        history_text_frame = ttk.Frame(history_frame)
        history_text_frame.pack(fill="both", expand=True)

        history_scrollbar = ttk.Scrollbar(history_text_frame, orient="vertical")
        history_scrollbar.pack(side="right", fill="y")

        history_text = tk.Text(
            history_text_frame,
            wrap="word",
            font=("Segoe UI", 9),
            state="disabled",
            yscrollcommand=history_scrollbar.set,
        )
        history_text.pack(side="left", fill="both", expand=True)
        self._bind_readonly_text_widget(history_text)
        history_scrollbar.configure(command=history_text.yview)
        self.ui_history_text = history_text

        desc = tk.Text(tab_desc, wrap="word", font=("Segoe UI", 9))
        desc.pack(fill="both", expand=True, padx=6, pady=6)
        desc.insert(
            "1.0",
            "LocalSTT portable\n\n"
            "Hotkeys (bound to physical Q / W / E key positions, independent of layout):\n"
            "Ctrl+Shift+Q - start/stop recording\n"
            "Ctrl+Shift+W - transcribe last recording\n"
            "Ctrl+Shift+E - exit\n\n"
            "In Russian layout these are the same physical keys where Й / Ц / У are printed.\n\n"
            "Default mode transcribes the whole recorded WAV after stop.\n"
            "An experimental live-overlap mode is still available through config if you want to compare it later.\n\n"
            "Set a preferred transcription language in settings to avoid language guessing.\n\n"
            "How it works:\n"
            "1) Start recording with a hotkey.\n"
            "2) In stable mode, after stop the full WAV is sent to faster-whisper.\n"
            "3) Recognized text is pasted into the active window.\n\n"
            "The Logs tab shows progress in real time.\n"
            "The History tab keeps the last 10 transcription results."
        )
        desc.configure(state="disabled")
        self._bind_readonly_text_widget(desc)

        mic_frame = ttk.Frame(tab_mic, padding=6)
        mic_frame.pack(fill="both", expand=True)

        ttk.Label(mic_frame, text="Input device:").pack(anchor="w")
        self.ui_mic_var = tk.StringVar()
        mic_combo = ttk.Combobox(mic_frame, textvariable=self.ui_mic_var, state="readonly")
        mic_combo.pack(fill="x", pady=(2, 6))
        self.ui_mic_combo = mic_combo

        row = ttk.Frame(mic_frame)
        row.pack(fill="x", pady=(0, 6))
        ttk.Button(row, text="Refresh", command=self._refresh_mic_devices).pack(side="left")
        ttk.Button(row, text="Select", command=self._apply_selected_mic).pack(side="left", padx=(6, 0))

        self.ui_mic_level_var = tk.DoubleVar(value=0.0)
        ttk.Label(mic_frame, text="Input level:").pack(anchor="w", pady=(6, 2))
        ttk.Progressbar(mic_frame, maximum=100.0, variable=self.ui_mic_level_var).pack(fill="x")

        self.ui_mic_test_btn = ttk.Button(mic_frame, text="Test microphone", command=self._toggle_mic_test)
        self.ui_mic_test_btn.pack(fill="x", pady=(8, 6))

        self.ui_vad_var = tk.BooleanVar(value=self.config.vad_filter)
        self.ui_restore_clipboard_var = tk.BooleanVar(value=self.config.restore_clipboard)
        self.ui_model_var = tk.StringVar(value=_model_display_value(self.config.model_size))
        self.ui_language_var = tk.StringVar(value=_language_display_value(self.config.transcription_language))

        ttk.Label(mic_frame, text="Speech model:").pack(anchor="w", pady=(8, 2))
        model_combo = ttk.Combobox(
            mic_frame,
            textvariable=self.ui_model_var,
            values=[_model_display_value(code) for code in self._available_model_codes()],
            state="readonly",
        )
        model_combo.pack(fill="x")
        self.ui_model_combo = model_combo
        ttk.Label(
            mic_frame,
            text="Only locally downloaded models are listed. Switching reloads the model and can take a few seconds.",
        ).pack(anchor="w", pady=(2, 6))

        ttk.Label(mic_frame, text="Preferred transcription language:").pack(anchor="w", pady=(8, 2))
        language_combo = ttk.Combobox(
            mic_frame,
            textvariable=self.ui_language_var,
            values=[label for label, _code in COMMON_LANGUAGE_OPTIONS],
        )
        language_combo.pack(fill="x")
        self.ui_language_combo = language_combo
        ttk.Label(
            mic_frame,
            text="Use Auto detect or type a Whisper language code such as en, ru, es.",
        ).pack(anchor="w", pady=(2, 6))

        def _apply_simple_settings() -> None:
            selected_language = _normalize_transcription_language(
                self.ui_language_var.get() if self.ui_language_var is not None else self.config.transcription_language,
                fallback="",
            )
            if not selected_language:
                self._set_status("Unsupported language code")
                self._show_popup("UNSUPPORTED LANGUAGE CODE", bg="#9a1b1b")
                return

            selected_model = _normalize_model_size(
                self.ui_model_var.get() if self.ui_model_var is not None else self.config.model_size,
                fallback="",
            )
            if not selected_model:
                self._set_status("Unsupported model")
                self._show_popup("UNSUPPORTED MODEL", bg="#9a1b1b")
                return

            vad_filter = bool(self.ui_vad_var.get())
            restore_clipboard = bool(self.ui_restore_clipboard_var.get())
            if self.ui_language_var is not None:
                self.ui_language_var.set(_language_display_value(selected_language))
            if self.ui_model_var is not None:
                self.ui_model_var.set(_model_display_value(selected_model))
            self._dispatch_ui_action(
                "save_settings",
                lambda: self._apply_user_settings(
                    vad_filter=vad_filter,
                    restore_clipboard=restore_clipboard,
                    transcription_language=selected_language,
                    model_size=selected_model,
                ),
            )

        ttk.Checkbutton(mic_frame, text="VAD filter", variable=self.ui_vad_var).pack(anchor="w")
        ttk.Checkbutton(
            mic_frame,
            text="Restore clipboard",
            variable=self.ui_restore_clipboard_var,
        ).pack(anchor="w")
        ttk.Button(mic_frame, text="Save settings", command=_apply_simple_settings).pack(fill="x", pady=(6, 0))

        recovery_frame = ttk.LabelFrame(container, text="Recovery actions", padding=6)
        recovery_frame.pack(fill="x", pady=(6, 0))

        ttk.Label(
            recovery_frame,
            text="Cancel the active job or recover the last paste safely.",
        ).pack(anchor="w", pady=(0, 6))

        cancel_btn = ttk.Button(
            recovery_frame,
            text="Cancel processing",
            command=lambda: self._dispatch_ui_action("cancel_transcription", self.cancel_transcription),
        )
        cancel_btn.pack(fill="x")
        self.ui_cancel_transcription_btn = cancel_btn

        repeat_btn = ttk.Button(
            recovery_frame,
            text="Re-paste last text",
            command=lambda: self._dispatch_ui_action("repeat_last_paste", self.repeat_last_paste),
        )
        repeat_btn.pack(fill="x", pady=(6, 0))
        self.ui_repeat_paste_btn = repeat_btn

        undo_btn = ttk.Button(
            recovery_frame,
            text="Undo last paste",
            command=lambda: self._dispatch_ui_action("undo_last_paste", self.undo_last_paste),
        )
        undo_btn.pack(fill="x", pady=(6, 0))
        self.ui_undo_paste_btn = undo_btn

        self.ui_status_var = tk.StringVar(value="Done")
        ttk.Label(container, textvariable=self.ui_status_var).pack(fill="x", pady=(6, 0))

        self._refresh_mic_devices()
        self._refresh_history_view()
        self._update_recovery_buttons()
        notebook.select(tab_logs)

        def _on_close() -> None:
            self.shutdown()

        root.protocol("WM_DELETE_WINDOW", _on_close)
        self._ui_poll()
        root.mainloop()

    def shutdown(self) -> None:
        if self.stop_event.is_set():
            return
        logging.info("Shutdown requested")
        self.stop_event.set()
        self._show_popup("LOCALSTT STOPPED", bg="#5d2f87")

        try:
            if self.is_recording:
                self.stop_recording()
        except Exception:
            logging.exception("Error while stopping recording")

        self._stop_mic_test()

        try:
            self.hotkeys.stop()
        except Exception:
            pass

        if self.ui_root is not None:
            try:
                if self.ui_toast_window is not None:
                    self.ui_toast_window.withdraw()
                self.ui_root.after(0, self.ui_root.destroy)
            except Exception:
                pass

    def run(self) -> None:
        logging.info("LocalSTT started")
        logging.info("Hotkeys: Ctrl+Shift + physical Q/W/E keys (same positions as Й/Ц/У)")
        self.hotkeys.start()
        self._build_ui()
        logging.info("LocalSTT stopped")


def main() -> None:
    offline_flag = os.environ.get("LOCALSTT_OFFLINE_ONLY", "1").strip().lower()
    offline_only = offline_flag in {"1", "true", "yes", "on"}

    restore_clipboard_flag = os.environ.get("LOCALSTT_RESTORE_CLIPBOARD", "1").strip().lower()
    restore_clipboard = restore_clipboard_flag in {"1", "true", "yes", "on"}

    chunk_duration_raw = os.environ.get("LOCALSTT_CHUNK_SEC", "2.0")
    try:
        chunk_duration = max(0.25, float(chunk_duration_raw))
    except ValueError:
        chunk_duration = 2.0

    chunk_overlap_raw = os.environ.get("LOCALSTT_CHUNK_OVERLAP_SEC", "0.5")
    try:
        chunk_overlap = max(0.0, float(chunk_overlap_raw))
    except ValueError:
        chunk_overlap = 0.5

    transcription_language = _normalize_transcription_language(os.environ.get("LOCALSTT_LANGUAGE", "auto"))
    transcription_mode = _normalize_transcription_mode(os.environ.get("LOCALSTT_MODE", "full-file"))

    config = AppConfig(
        model_size=os.environ.get("LOCALSTT_MODEL", "small"),
        model_path=os.environ.get("LOCALSTT_MODEL_PATH", "models/faster-whisper-small"),
        offline_only=offline_only,
        compute_type=os.environ.get("LOCALSTT_COMPUTE", "int8"),
        device=os.environ.get("LOCALSTT_DEVICE", "auto"),
        restore_clipboard=restore_clipboard,
        input_device=os.environ.get("LOCALSTT_INPUT_DEVICE"),
        transcription_mode=transcription_mode,
        chunk_duration_sec=chunk_duration,
        chunk_overlap_sec=chunk_overlap,
        transcription_language=transcription_language,
    )

    if "LOCALSTT_MODEL_PATH" not in os.environ:
        config.model_path = _model_path_for_size(config.model_size)

    app = LocalSTTApp(config)
    app.run()


if __name__ == "__main__":
    main()
