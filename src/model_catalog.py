from typing import Any


MODEL_ORDER = ("tiny", "small", "medium")
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
}

MODEL_SIZE_TO_LABEL = {size: spec["label"] for size, spec in MODEL_SPECS.items()}
MODEL_LABEL_TO_SIZE = {label: size for size, label in MODEL_SIZE_TO_LABEL.items()}


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