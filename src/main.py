import logging
import os
import queue
import sys
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pyautogui
import sounddevice as sd
from pynput import keyboard
from pynput.keyboard import Controller

from app_settings import (
    AppConfig,
    AppSettingsMixin,
    _normalize_transcription_language,
    _normalize_transcription_mode,
    is_assemblyai_model,
    model_label,
    model_relative_path,
    normalize_model_size,
)
from app_ui import AppUiMixin
from audio_runtime import AudioRuntimeMixin
from transcription_runtime import TranscriptionRuntimeMixin


CTRL_KEYS = {keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r}
SHIFT_KEYS = {keyboard.Key.shift, keyboard.Key.shift_l, keyboard.Key.shift_r}


class QueueLogHandler(logging.Handler):
    def __init__(self, q: queue.Queue[str]) -> None:
        super().__init__()
        self.q = q

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.q.put(self.format(record))
        except Exception:
            pass


class LocalSTTApp(AppUiMixin, AppSettingsMixin, AudioRuntimeMixin, TranscriptionRuntimeMixin):
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.config.model_size = normalize_model_size(self.config.model_size)
        if not str(self.config.model_path).strip():
            self.config.model_path = model_relative_path(self.config.model_size)

        if getattr(sys, "frozen", False):
            self.project_root = Path(sys.executable).resolve().parent
        else:
            self.project_root = Path(__file__).resolve().parent.parent

        local_app_data = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "LocalSTT"
        self.recordings_dir = local_app_data / "recordings"
        self.logs_dir = local_app_data / "logs"
        if getattr(sys, "frozen", False):
            self.models_dir = self.project_root / "_internal" / "models"
        else:
            self.models_dir = local_app_data / "models"
        self.models_dir_label = str(self.models_dir)
        self.settings_file = local_app_data / "settings.json"
        self.history_file = local_app_data / "history.json"
        self.recordings_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)

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
        self.model_download_in_progress = False
        self.model_status_cache: dict[str, dict[str, Any]] = {}
        self.startup_model_notice: str | None = None

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
        self.ui_model_current_var = None
        self.ui_model_details_var = None
        self.ui_model_hint_var = None
        self.ui_model_action_btn = None
        self.ui_assemblyai_frame = None
        self.ui_assemblyai_key_var = None
        self.ui_assemblyai_key_entry = None
        self.ui_assemblyai_hint_var = None

        self._load_settings_from_file()
        self._sync_model_config()
        self._load_history_from_file()

        if self.config.offline_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

        self._refresh_model_status_cache()
        if is_assemblyai_model(self.config.model_size):
            self.model = None
            if not self._assemblyai_api_key():
                self.startup_model_notice = "AssemblyAI selected. Enter your API key in the Microphone tab."
        else:
            try:
                self.model = self._load_model_instance(self.config.model_size, self.config.model_path)
            except FileNotFoundError:
                logging.exception("Configured model is unavailable during startup")
                if normalize_model_size(self.config.model_size) != "small":
                    missing_label = model_label(self.config.model_size)
                    self.startup_model_notice = f"{missing_label} was unavailable. Switched back to Small (default)."
                    self.config.model_size = "small"
                    self.config.model_path = model_relative_path("small")
                    self.model = self._load_model_instance(self.config.model_size, self.config.model_path)
                    self._save_settings_to_file()
                else:
                    raise
        self._refresh_model_status_cache()

        self.input_device = self._resolve_input_device(self.config.input_device)
        self._log_audio_input_info()

        self.hotkey_actions_by_vk: dict[int, tuple[str, Any]] = {
            0x41: ("cancel_recording", self.cancel_recording),
            0x51: ("toggle_recording", self.toggle_recording),
            0x57: ("transcribe_last", self.transcribe_last_file),
            0x45: ("shutdown", self.shutdown),
        }
        self.hotkey_actions_by_scan: dict[int, tuple[str, Any]] = {
            0x1E: ("cancel_recording", self.cancel_recording),
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
            import time

            elapsed_sec = max(0, int(time.perf_counter() - self.recording_started_at))
            timer_text = self._format_popup_elapsed(elapsed_sec)
            base_text = self.ui_popup_base_text or "RECORDING..."
            self.ui_toast_label.configure(text=f"{base_text} {timer_text}")
            return

        if self.ui_popup_timer_mode == "transcribing":
            if not self.is_transcribing or self.transcription_started_at is None:
                return
            import time

            elapsed_sec = max(0, int(time.perf_counter() - self.transcription_started_at))
            timer_text = self._format_popup_elapsed(elapsed_sec, self.transcription_target_duration_sec)
            base_text = self.ui_popup_base_text or "TRANSCRIBING..."
            self.ui_toast_label.configure(text=f"{base_text} {timer_text}")

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
        logging.info("Hotkeys: Ctrl+Shift + physical A/Q/W/E keys (same positions as Р¤/Р™/Р¦/РЈ)")
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
    configured_model_size = normalize_model_size(os.environ.get("LOCALSTT_MODEL", "small"))

    config = AppConfig(
        model_size=configured_model_size,
        model_path=os.environ.get("LOCALSTT_MODEL_PATH", model_relative_path(configured_model_size)),
        offline_only=offline_only,
        compute_type=os.environ.get("LOCALSTT_COMPUTE", "int8"),
        device=os.environ.get("LOCALSTT_DEVICE", "auto"),
        restore_clipboard=restore_clipboard,
        input_device=os.environ.get("LOCALSTT_INPUT_DEVICE"),
        transcription_mode=transcription_mode,
        chunk_duration_sec=chunk_duration,
        chunk_overlap_sec=chunk_overlap,
        transcription_language=transcription_language,
        assemblyai_api_key=os.environ.get("LOCALSTT_ASSEMBLYAI_API_KEY", ""),
    )

    if "LOCALSTT_MODEL_PATH" not in os.environ:
        config.model_path = model_relative_path(config.model_size)

    app = LocalSTTApp(config)
    app.run()


if __name__ == "__main__":
    main()