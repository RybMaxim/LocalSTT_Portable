import gc
import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download
from huggingface_hub.utils import are_progress_bars_disabled, disable_progress_bars, enable_progress_bars


ASSEMBLYAI_MODEL = "assemblyai"
ASSEMBLYAI_SPEECH_MODELS = ("universal-3-pro", "universal-2")
LOCAL_MODEL_ORDER = ("tiny", "small", "medium")

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

MODEL_ORDER = LOCAL_MODEL_ORDER + (ASSEMBLYAI_MODEL,)
MODEL_REQUIRED_FILES = ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt")

MODEL_SPECS: dict[str, dict[str, Any]] = {
    "tiny": {
        "label": "Tiny",
        "repo_id": "Systran/faster-whisper-tiny",
        "folder_name": "faster-whisper-tiny",
        "relative_path": "models/faster-whisper-tiny",
        "expected_size_bytes": 78207736,
        "description": "Fastest model, lowest recognition accuracy.",
    },
    "small": {
        "label": "Small (default)",
        "repo_id": "Systran/faster-whisper-small",
        "folder_name": "faster-whisper-small",
        "relative_path": "models/faster-whisper-small",
        "expected_size_bytes": 486216494,
        "description": "Balanced speed and quality. Included with the app by default.",
    },
    "medium": {
        "label": "Medium",
        "repo_id": "Systran/faster-whisper-medium",
        "folder_name": "faster-whisper-medium",
        "relative_path": "models/faster-whisper-medium",
        "expected_size_bytes": 1530575865,
        "description": "Higher quality model with the largest download size.",
    },
    ASSEMBLYAI_MODEL: {
        "label": "AssemblyAI",
        "repo_id": "",
        "folder_name": "assemblyai",
        "relative_path": "",
        "expected_size_bytes": 0,
        "description": "Cloud transcription via AssemblyAI using Universal-3 Pro with Universal-2 fallback.",
    },
}

MODEL_SIZE_TO_LABEL = {size: spec["label"] for size, spec in MODEL_SPECS.items()}
MODEL_LABEL_TO_SIZE = {label: size for size, label in MODEL_SIZE_TO_LABEL.items()}
COMMON_MODEL_OPTIONS: list[tuple[str, str]] = [(MODEL_SIZE_TO_LABEL[size], size) for size in MODEL_ORDER]
MODEL_LABEL_TO_CODE = {label: code for label, code in COMMON_MODEL_OPTIONS}
MODEL_CODE_TO_LABEL = {code: label for label, code in COMMON_MODEL_OPTIONS}
MODEL_CODE_TO_PATH = {size: str(spec["relative_path"]) for size, spec in MODEL_SPECS.items()}

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


def normalize_model_size(value: Any, fallback: str = "small") -> str:
    if value is None:
        return fallback

    raw = str(value).strip()
    if not raw:
        return fallback

    lowered = raw.casefold()
    if lowered in MODEL_SPECS:
        return lowered

    for size, label in MODEL_SIZE_TO_LABEL.items():
        if lowered == label.casefold():
            return size

    if lowered.startswith("small"):
        return "small"
    if lowered.startswith("medium"):
        return "medium"
    if lowered.startswith("tiny"):
        return "tiny"

    return fallback


def model_label(model_size: str) -> str:
    normalized = normalize_model_size(model_size)
    return str(MODEL_SPECS[normalized]["label"])


def model_repo_id(model_size: str) -> str:
    normalized = normalize_model_size(model_size)
    return str(MODEL_SPECS[normalized]["repo_id"])


def model_storage_folder_name(model_size: str) -> str:
    normalized = normalize_model_size(model_size)
    return str(MODEL_SPECS[normalized]["folder_name"])


def model_relative_path(model_size: str) -> str:
    normalized = normalize_model_size(model_size)
    return str(MODEL_SPECS[normalized]["relative_path"])


def model_description(model_size: str) -> str:
    normalized = normalize_model_size(model_size)
    return str(MODEL_SPECS[normalized]["description"])


def expected_model_size_bytes(model_size: str) -> int:
    normalized = normalize_model_size(model_size)
    return int(MODEL_SPECS[normalized]["expected_size_bytes"])


def format_model_size(size_bytes: int) -> str:
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def is_assemblyai_model(value: Any) -> bool:
    return normalize_model_size(value, fallback="") == ASSEMBLYAI_MODEL


def uses_local_model(value: Any) -> bool:
    return normalize_model_size(value, fallback="") in LOCAL_MODEL_ORDER


def _normalize_model_size(value: Any, fallback: str = "small") -> str:
    if value is None:
        return fallback

    raw = str(value).strip()
    if not raw:
        return fallback
    if raw in MODEL_LABEL_TO_CODE:
        return MODEL_LABEL_TO_CODE[raw]
    if raw in MODEL_LABEL_TO_SIZE:
        return MODEL_LABEL_TO_SIZE[raw]

    normalized_catalog_value = normalize_model_size(raw, fallback="")
    if normalized_catalog_value:
        return normalized_catalog_value

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
    if normalized == ASSEMBLYAI_MODEL:
        return ASSEMBLYAI_MODEL
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
    assemblyai_api_key: str = ""


class AppSettingsMixin:
    def _runtime_roots(self) -> list[Path]:
        roots: list[Path] = []
        if getattr(os.sys, "frozen", False):
            meipass = getattr(os.sys, "_MEIPASS", None)
            if meipass:
                roots.append(Path(meipass))
            roots.append(self.project_root / "_internal")
        roots.append(self.project_root)
        return roots

    def _resolve_existing_path(self, relative_path: str) -> Path | None:
        rel = Path(relative_path)
        for root in self._runtime_roots():
            candidate = root / rel
            if candidate.exists():
                return candidate
        return None

    def _resolve_model_source(self, model_path_value: str | None = None, model_size_value: str | None = None):
        configured_model_path = model_path_value if model_path_value is not None else self.config.model_path
        configured_model_size = model_size_value if model_size_value is not None else self.config.model_size

        model_path = Path(configured_model_path)
        if not model_path.is_absolute():
            found = self._resolve_existing_path(str(model_path))
            if found is not None:
                model_path = found
            else:
                model_path = self.project_root / model_path

        if model_path.exists():
            return str(model_path), True

        if self.config.offline_only:
            raise FileNotFoundError(
                f"Offline model not found at '{model_path}'. Download model once before startup."
            )

        return configured_model_size, False

    def _sync_model_config(self) -> None:
        fallback_model_size = _model_size_from_path(self.config.model_path, fallback=self.config.model_size or "small")
        self.config.model_size = _normalize_model_size(self.config.model_size, fallback=fallback_model_size)
        if is_assemblyai_model(self.config.model_size):
            self.config.model_path = model_relative_path(self.config.model_size)
            return

        configured_model_size = _model_size_from_path(self.config.model_path, fallback="")
        if configured_model_size:
            self.config.model_path = _model_path_for_size(self.config.model_size)
        elif not str(self.config.model_path).strip():
            self.config.model_path = _model_path_for_size(self.config.model_size)

    def _assemblyai_api_key(self) -> str:
        return str(getattr(self.config, "assemblyai_api_key", "") or "").strip()

    def _ensure_active_provider_ready(self) -> bool:
        if is_assemblyai_model(self.config.model_size) and not self._assemblyai_api_key():
            self._set_status("Enter your AssemblyAI API key first")
            self._show_popup("ASSEMBLYAI API KEY REQUIRED", bg="#9a1b1b")
            return False
        return True

    def _format_model_status_text(self, prefix: str, status: dict[str, Any]) -> str:
        if is_assemblyai_model(status["size"]):
            return f"{prefix}: {status['label']} | {status['status_label']}"

        size_text = format_model_size(int(status["size_bytes"]))
        return f"{prefix}: {status['label']} | {size_text} | {status['status_label']}"

    def _update_assemblyai_settings_visibility(self) -> None:
        frame = getattr(self, "ui_assemblyai_frame", None)
        if frame is None:
            return

        is_visible = is_assemblyai_model(self._selected_model_size())
        manager = str(frame.winfo_manager())
        if is_visible and not manager:
            frame.pack(fill="x", pady=(2, 8))
        elif not is_visible and manager:
            frame.pack_forget()

        hint_var = getattr(self, "ui_assemblyai_hint_var", None)
        if hint_var is not None:
            if not is_visible:
                hint_var.set("")
            elif self._assemblyai_api_key():
                hint_var.set(
                    "AssemblyAI uses Universal-3 Pro with Universal-2 fallback and works in full-file mode only."
                )
            else:
                hint_var.set("Enter your own AssemblyAI API key to enable cloud transcription.")

    def _switch_to_assemblyai(self) -> bool:
        api_key = self._assemblyai_api_key()
        if not api_key:
            self._set_status("Enter your AssemblyAI API key first")
            self._show_popup("ASSEMBLYAI API KEY REQUIRED", bg="#9a1b1b")
            return False

        previous_model = getattr(self, "model", None)
        self.model = None
        self.config.model_size = ASSEMBLYAI_MODEL
        self.config.model_path = model_relative_path(ASSEMBLYAI_MODEL)
        self._save_settings_to_file()

        try:
            if previous_model is not None:
                del previous_model
                gc.collect()
        except Exception:
            pass

        self._refresh_model_status_cache()
        self._refresh_model_ui_async()
        self._set_model_ui_value(self.config.model_size)
        self._hide_popup()
        self._set_status("AssemblyAI is active")
        self._show_popup("ASSEMBLYAI READY", bg="#1e6b2d")
        return True

    def _set_model_ui_value(self, model_size: str) -> None:
        if self.ui_root is None or self.ui_model_var is None:
            return

        label = MODEL_SIZE_TO_LABEL.get(normalize_model_size(model_size), model_label(model_size))
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
        return self._load_model_instance(self.config.model_size, self.config.model_path)

    def _apply_user_settings(
        self,
        *,
        vad_filter: bool,
        restore_clipboard: bool,
        transcription_language: str,
        model_size: str,
        assemblyai_api_key: str,
    ) -> None:
        previous_model_size = self.config.model_size
        previous_model_path = self.config.model_path
        previous_api_key = self._assemblyai_api_key()

        self.config.vad_filter = vad_filter
        self.config.restore_clipboard = restore_clipboard
        self.config.transcription_language = transcription_language
        self.config.assemblyai_api_key = str(assemblyai_api_key or "").strip()

        normalized_model_size = _normalize_model_size(model_size, fallback="")
        if not normalized_model_size:
            self._set_status("Unsupported model")
            self._show_popup("UNSUPPORTED MODEL", bg="#9a1b1b")
            self._set_model_ui_value(previous_model_size)
            self.config.assemblyai_api_key = previous_api_key
            return

        if is_assemblyai_model(normalized_model_size) and not self._assemblyai_api_key():
            self._set_status("Enter your AssemblyAI API key")
            self._show_popup("ASSEMBLYAI API KEY REQUIRED", bg="#9a1b1b")
            self.config.assemblyai_api_key = previous_api_key
            self._refresh_model_ui_async()
            return

        model_changed = normalized_model_size != previous_model_size
        if not model_changed:
            self._save_settings_to_file()
            self._set_status("Settings saved")
            self._refresh_model_ui_async()
            return

        if self.model_download_in_progress:
            self._set_model_ui_value(previous_model_size)
            self._set_status("Wait for the current model download to finish")
            self._show_popup("MODEL DOWNLOAD IN PROGRESS", bg="#7d5a11")
            return

        if self.is_recording or self.is_transcribing:
            self._save_settings_to_file()
            self._set_model_ui_value(previous_model_size)
            self._set_status("Stop the active job before switching model")
            self._show_popup("STOP ACTIVE JOB FIRST", bg="#9a1b1b")
            return

        if is_assemblyai_model(normalized_model_size):
            self._set_status("Enabling AssemblyAI...")
            self._show_popup("ENABLING ASSEMBLYAI...", bg="#0d5f8a", persistent=True)
            self._switch_to_assemblyai()
            return

        target_model_label = model_label(normalized_model_size)
        self._set_status(f"Loading {target_model_label}...")
        self._show_popup("LOADING MODEL...", bg="#0d5f8a", persistent=True)

        try:
            self._apply_selected_model(normalized_model_size)
            new_model = self._load_whisper_model()
        except Exception:
            self.config.model_size = previous_model_size
            self.config.model_path = previous_model_path
            self._save_settings_to_file()
            self._set_model_ui_value(previous_model_size)
            self._refresh_model_status_cache()
            self._refresh_model_ui_async()
            self._hide_popup()
            logging.exception("Failed to switch model")
            self._set_status(f"Failed to load {target_model_label}")
            self._show_popup("MODEL LOAD FAILED", bg="#9a1b1b")
            return

        previous_model = self.model
        self.model = new_model

        try:
            del previous_model
            gc.collect()
        except Exception:
            pass

        self._save_settings_to_file()
        self._refresh_model_status_cache()
        self._refresh_model_ui_async()
        self._set_model_ui_value(self.config.model_size)
        self._hide_popup()
        self._set_status(f"Model switched to {target_model_label}")
        self._show_popup("MODEL READY", bg="#1e6b2d")

    def _load_model_instance(self, model_size: str, model_path: str) -> WhisperModel:
        if is_assemblyai_model(model_size):
            raise ValueError("AssemblyAI does not use a local Whisper model instance")

        resolved_size = normalize_model_size(model_size)
        resolved_path = str(model_path).strip() or model_relative_path(resolved_size)
        model_source, local_files_only = self._resolve_model_source(resolved_path, resolved_size)
        logging.info("Loading faster-whisper model from: %s", model_source)
        model = WhisperModel(
            model_source,
            device=self.config.device,
            compute_type=self.config.compute_type,
            local_files_only=local_files_only,
        )
        logging.info("Model loaded")
        return model

    def _model_storage_path(self, model_size: str) -> Path:
        return self.models_dir / model_storage_folder_name(model_size)

    def _config_model_path(self, model_size: str, model_path: Path | None) -> str:
        normalized = normalize_model_size(model_size)
        relative_path = model_relative_path(normalized)
        if model_path is None:
            return relative_path

        candidate_roots = self._runtime_roots()
        relative_candidate = Path(relative_path)
        for root in candidate_roots:
            if (root / relative_candidate) == model_path:
                return relative_path
            try:
                if (root / relative_candidate).resolve() == model_path.resolve():
                    return relative_path
            except OSError:
                continue

        return str(model_path)

    def _is_complete_model_folder(self, model_path: Path) -> bool:
        return model_path.is_dir() and all((model_path / filename).exists() for filename in MODEL_REQUIRED_FILES)

    def _directory_size_bytes(self, root: Path) -> int:
        total_size = 0
        for path in root.rglob("*"):
            if path.is_file():
                try:
                    total_size += path.stat().st_size
                except OSError:
                    logging.warning("Could not read file size: %s", path)
        return total_size

    def _resolve_runtime_model_path(self, model_size: str) -> tuple[Path | None, str]:
        if is_assemblyai_model(model_size):
            source = "assemblyai-ready" if self._assemblyai_api_key() else "assemblyai-key-missing"
            return None, source

        resolved = self._resolve_existing_path(model_relative_path(model_size))
        if resolved is None or not self._is_complete_model_folder(resolved):
            return None, "missing"
        if normalize_model_size(model_size) == "small":
            return resolved, "bundled"
        return resolved, "local"

    def _model_source_label(self, source: str) -> str:
        labels = {
            "bundled": "Included with the app by default",
            "downloaded": f"Downloaded to {self.models_dir_label}",
            "local": "Found in a local models folder",
            "missing": "Not downloaded yet",
            "assemblyai-ready": "Cloud API ready",
            "assemblyai-key-missing": "API key required",
        }
        return labels.get(source, source)

    def _build_model_status(self, model_size: str) -> dict[str, Any]:
        normalized = normalize_model_size(model_size)
        if is_assemblyai_model(normalized):
            source = "assemblyai-ready" if self._assemblyai_api_key() else "assemblyai-key-missing"
            return {
                "size": normalized,
                "label": model_label(normalized),
                "installed": bool(self._assemblyai_api_key()),
                "active": normalize_model_size(self.config.model_size) == normalized,
                "source": source,
                "status_label": self._model_source_label(source),
                "path": None,
                "storage_path": None,
                "config_path": model_relative_path(normalized),
                "size_bytes": 0,
            }

        downloaded_path = self._model_storage_path(normalized)
        runtime_path, runtime_source = self._resolve_runtime_model_path(normalized)
        installed_path: Path | None = None
        source = "missing"

        if normalized == "small":
            if runtime_path is not None:
                installed_path = runtime_path
                source = runtime_source
            elif self._is_complete_model_folder(downloaded_path):
                installed_path = downloaded_path
                source = "downloaded"
        else:
            if self._is_complete_model_folder(downloaded_path):
                installed_path = downloaded_path
                source = "downloaded"
            elif runtime_path is not None:
                installed_path = runtime_path
                source = runtime_source

        size_bytes = expected_model_size_bytes(normalized)
        if installed_path is not None:
            measured_size = self._directory_size_bytes(installed_path)
            if measured_size > 0:
                size_bytes = measured_size

        return {
            "size": normalized,
            "label": model_label(normalized),
            "installed": installed_path is not None,
            "active": normalize_model_size(self.config.model_size) == normalized,
            "source": source,
            "status_label": self._model_source_label(source),
            "path": installed_path,
            "storage_path": downloaded_path,
            "config_path": self._config_model_path(normalized, installed_path),
            "size_bytes": size_bytes,
        }

    def _refresh_model_status_cache(self) -> None:
        self.model_status_cache = {model_size: self._build_model_status(model_size) for model_size in MODEL_ORDER}

    def _model_status(self, model_size: str) -> dict[str, Any]:
        normalized = normalize_model_size(model_size)
        cached = self.model_status_cache.get(normalized)
        if cached is not None:
            return cached

        status = self._build_model_status(normalized)
        self.model_status_cache[normalized] = status
        return status

    def _selected_model_size(self) -> str:
        if self.ui_model_var is None:
            return normalize_model_size(self.config.model_size)
        raw_value = self.ui_model_var.get().strip()
        if raw_value in MODEL_LABEL_TO_SIZE:
            return MODEL_LABEL_TO_SIZE[raw_value]
        return normalize_model_size(raw_value, fallback=self.config.model_size)

    def _refresh_model_ui_async(self) -> None:
        self._refresh_model_status_cache()
        if self.ui_root is not None:
            try:
                self.ui_root.after(0, self._render_model_ui)
            except Exception:
                logging.exception("Failed to schedule model UI refresh")

    def _render_model_ui(self) -> None:
        if self.ui_model_var is None:
            return

        if not self.model_status_cache:
            self._refresh_model_status_cache()

        current_size = normalize_model_size(self.config.model_size)
        current_status = self._model_status(current_size)
        if self.ui_model_current_var is not None:
            self.ui_model_current_var.set(self._format_model_status_text("Current model", current_status))

        selected_size = self._selected_model_size()
        selected_label = MODEL_SIZE_TO_LABEL.get(selected_size, model_label(selected_size))
        if self.ui_model_var.get().strip() != selected_label:
            self.ui_model_var.set(selected_label)
            selected_size = self._selected_model_size()

        selected_status = self._model_status(selected_size)
        if self.ui_model_details_var is not None:
            self.ui_model_details_var.set(self._format_model_status_text("Selected", selected_status))

        self._update_assemblyai_settings_visibility()
        self._update_model_controls()

    def _update_model_controls(self) -> None:
        if self.ui_model_action_btn is None:
            return

        selected_status = self.model_status_cache.get(self._selected_model_size())
        if selected_status is None:
            return

        if self.ui_model_hint_var is not None:
            if self.model_download_in_progress:
                self.ui_model_hint_var.set("Downloading the selected model...")
            elif is_assemblyai_model(selected_status["size"]):
                if self._assemblyai_api_key():
                    self.ui_model_hint_var.set(
                        "AssemblyAI uses Universal-3 Pro with Universal-2 fallback and ignores live-overlap mode."
                    )
                else:
                    self.ui_model_hint_var.set("Enter your AssemblyAI API key below to enable cloud transcription.")
            elif self.is_recording or self.is_transcribing:
                self.ui_model_hint_var.set("Finish recording or transcription before changing the model.")
            else:
                self.ui_model_hint_var.set("")

        is_busy = self.is_recording or self.is_transcribing or self.model_download_in_progress
        if self.model_download_in_progress:
            button_text = "Downloading model..."
            disabled = True
        elif is_assemblyai_model(selected_status["size"]):
            if selected_status["active"]:
                button_text = "AssemblyAI is active"
                disabled = True
            elif not self._assemblyai_api_key():
                button_text = "Enter API key below"
                disabled = True
            else:
                button_text = "Use AssemblyAI"
                disabled = is_busy
        elif selected_status["active"]:
            button_text = f"{selected_status['label']} is active"
            disabled = True
        elif not selected_status["installed"]:
            button_text = f"Download {selected_status['label']}"
            disabled = is_busy
        else:
            button_text = f"Use {selected_status['label']}"
            disabled = is_busy

        self.ui_model_action_btn.configure(text=button_text)
        if disabled:
            self.ui_model_action_btn.state(["disabled"])
        else:
            self.ui_model_action_btn.state(["!disabled"])

    def _ensure_model_change_allowed(self) -> bool:
        if self.model_download_in_progress:
            self._set_status("Model download already in progress")
            self._show_popup("MODEL DOWNLOAD IN PROGRESS", bg="#7d5a11")
            return False
        if self.is_recording:
            self._set_status("Stop recording before changing the model")
            self._show_popup("STOP RECORDING FIRST", bg="#7d5a11")
            return False
        if self.is_transcribing:
            self._set_status("Wait until transcription finishes before changing the model")
            self._show_popup("TRANSCRIPTION STILL RUNNING", bg="#7d5a11")
            return False
        return True

    def _download_model(self, model_size: str) -> bool:
        if is_assemblyai_model(model_size):
            self._set_status("AssemblyAI does not require a local download")
            self._show_popup("ASSEMBLYAI USES CLOUD API", bg="#7d5a11")
            return False

        normalized = normalize_model_size(model_size)
        model_name = model_label(normalized)
        model_size_text = format_model_size(expected_model_size_bytes(normalized))
        target_dir = self._model_storage_path(normalized)
        target_dir.mkdir(parents=True, exist_ok=True)

        self.model_download_in_progress = True
        self._refresh_model_ui_async()
        self._set_status(f"Downloading {model_name} ({model_size_text})...")
        self._show_popup("DOWNLOADING MODEL...", bg="#0d5f8a", persistent=True)
        logging.info("Downloading model %s from %s to %s", model_name, model_repo_id(normalized), target_dir)

        previous_offline = os.environ.pop("HF_HUB_OFFLINE", None)
        progress_bars_were_disabled = are_progress_bars_disabled()
        try:
            disable_progress_bars()
            snapshot_download(
                repo_id=model_repo_id(normalized),
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            if not self._is_complete_model_folder(target_dir):
                raise RuntimeError(f"Downloaded model is incomplete: {target_dir}")
            logging.info("Model download completed: %s", target_dir)
            self._set_status(f"Downloaded {model_name}")
            self._show_popup("MODEL DOWNLOADED", bg="#1e6b2d")
            return True
        except Exception:
            logging.exception("Failed to download model: %s", model_name)
            self._set_status(f"Download failed for {model_name}")
            self._show_popup("MODEL DOWNLOAD FAILED", bg="#9a1b1b")
            return False
        finally:
            if previous_offline is not None:
                os.environ["HF_HUB_OFFLINE"] = previous_offline
            elif self.config.offline_only:
                os.environ["HF_HUB_OFFLINE"] = "1"
            if not progress_bars_were_disabled:
                enable_progress_bars()
            self.model_download_in_progress = False
            self._refresh_model_ui_async()

    def _activate_model(self, model_size: str) -> bool:
        if is_assemblyai_model(model_size):
            return self._switch_to_assemblyai()

        normalized = normalize_model_size(model_size)
        status = self._model_status(normalized)
        if not status["installed"]:
            self._set_status(f"{status['label']} is not available locally")
            self._show_popup("MODEL NOT AVAILABLE", bg="#9a1b1b")
            return False

        model_name = status["label"]
        model_path = str(status["config_path"])
        self._set_status(f"Loading {model_name}...")
        self._show_popup("LOADING MODEL...", bg="#0d5f8a", persistent=True)

        try:
            new_model = self._load_model_instance(normalized, model_path)
        except Exception:
            logging.exception("Failed to activate model: %s", model_name)
            self._set_status(f"Failed to load {model_name}")
            self._show_popup("MODEL LOAD FAILED", bg="#9a1b1b")
            return False

        previous_model = self.model
        self.model = new_model
        self.config.model_size = normalized
        self.config.model_path = model_path
        self._save_settings_to_file()

        try:
            del previous_model
            gc.collect()
        except Exception:
            pass

        self._refresh_model_ui_async()
        self._set_status(f"Active model: {model_name}")
        self._show_popup("MODEL READY", bg="#1e6b2d")
        return True

    def _handle_selected_model_action(self) -> None:
        if not self._ensure_model_change_allowed():
            return

        selected_size = self._selected_model_size()
        selected_status = self._model_status(selected_size)
        if not selected_status["installed"]:
            if not self._download_model(selected_size):
                return
            self._refresh_model_status_cache()
            selected_status = self._model_status(selected_size)

        if selected_status["active"]:
            self._set_status(f"{selected_status['label']} is already active")
            self._show_popup("MODEL ALREADY ACTIVE", bg="#7d5a11")
            return

        self._activate_model(selected_size)

    def _refresh_model_statuses(self) -> None:
        self._refresh_model_status_cache()
        self._set_status("Model list refreshed")
        self._refresh_model_ui_async()

    def _disable_combobox_mousewheel(self, widget, scroll_handler=None) -> None:
        def _handle(event):
            if callable(scroll_handler):
                scroll_handler(event)
            return "break"

        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            widget.bind(sequence, _handle)

    def _bind_mousewheel_scroll(self, widget, scroll_handler) -> None:
        if widget.winfo_class() not in {"TCombobox", "Combobox"}:
            for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
                widget.bind(sequence, scroll_handler, add="+")
        for child in widget.winfo_children():
            self._bind_mousewheel_scroll(child, scroll_handler)

    def _load_settings_from_file(self) -> None:
        try:
            if not self.settings_file.exists():
                return
            with self.settings_file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            self.config.input_device = data.get("input_device", self.config.input_device)
            self.config.vad_filter = bool(data.get("vad_filter", self.config.vad_filter))
            self.config.restore_clipboard = bool(data.get("restore_clipboard", self.config.restore_clipboard))
            self.config.model_size = normalize_model_size(
                data.get("model_size", self.config.model_size),
                fallback=self.config.model_size,
            )
            self.config.assemblyai_api_key = str(data.get("assemblyai_api_key", self.config.assemblyai_api_key) or "").strip()

            saved_model_path = str(data.get("model_path", self.config.model_path)).strip()
            self.config.model_path = saved_model_path or model_relative_path(self.config.model_size)
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
                "assemblyai_api_key": self._assemblyai_api_key(),
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