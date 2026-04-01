import logging
import os
import queue
import re
import sys
import threading
import time
import wave
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import numpy as np
import pyautogui
import pyperclip
import sounddevice as sd
from pynput.keyboard import Key, Controller

import live_overlap_mode
from os_adapter import get_os_adapter


class TranscriptionCancelledError(Exception):
    pass


def _to_int_if_numeric(value):
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value


def _normalize_device_name(name: str) -> str:
    cleaned = (name or "").strip()
    marker = " (@"
    pos = cleaned.find(marker)
    if pos > 0:
        cleaned = cleaned[:pos].strip()
    cleaned = " ".join(cleaned.split())
    return cleaned


def prepare_audio_samples(audio: np.ndarray, sample_rate: int) -> tuple[np.ndarray, int]:
    if audio.size == 0:
        return np.array([], dtype=np.float32), sample_rate

    original_dtype = audio.dtype
    if audio.ndim > 1:
        if audio.shape[1] == 1:
            audio = audio[:, 0]
        else:
            audio = audio.reshape(-1, audio.shape[1]).mean(axis=1)

    audio = audio.astype(np.float32)
    if np.issubdtype(original_dtype, np.integer):
        info = np.iinfo(original_dtype)
        if np.issubdtype(original_dtype, np.unsignedinteger):
            midpoint = float(info.max) / 2.0
            if midpoint > 0:
                audio = (audio - midpoint) / midpoint
        else:
            max_abs = float(max(abs(info.min), info.max))
            if max_abs > 0:
                audio /= max_abs
    audio = np.clip(audio, -1.0, 1.0)

    target_rate = 16000
    if sample_rate != target_rate and audio.size > 0:
        src_x = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
        target_len = max(1, int(round(audio.shape[0] * (target_rate / float(sample_rate)))))
        dst_x = np.linspace(0.0, 1.0, num=target_len, endpoint=False)
        audio = np.interp(dst_x, src_x, audio).astype(np.float32)
        sample_rate = target_rate

    return audio.astype(np.float32), sample_rate


def load_audio_for_transcription(audio_path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(audio_path), "rb") as wav_file:
        channels = int(wav_file.getnchannels())
        sample_rate = int(wav_file.getframerate())
        sampwidth = int(wav_file.getsampwidth())
        raw = wav_file.readframes(wav_file.getnframes())

    dtype_map = {
        1: np.uint8,
        2: np.int16,
        4: np.int32,
    }
    dtype = dtype_map.get(sampwidth)
    if dtype is None:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth}")

    audio = np.frombuffer(raw, dtype=dtype)
    if channels > 1:
        audio = audio.reshape(-1, channels)

    return prepare_audio_samples(audio, sample_rate)


def _build_chunk_ranges(total_samples: int, chunk_samples: int, overlap_samples: int) -> list[tuple[int, int]]:
    if total_samples <= 0:
        return []

    overlap_samples = max(0, min(overlap_samples, chunk_samples - 1))
    step_samples = max(1, chunk_samples - overlap_samples)
    ranges: list[tuple[int, int]] = []
    start_idx = 0
    while start_idx < total_samples:
        end_idx = min(total_samples, start_idx + chunk_samples)
        ranges.append((start_idx, end_idx))
        if end_idx >= total_samples:
            break
        start_idx += step_samples
    return ranges


def _normalize_token_for_match(token: str) -> str:
    return re.sub(r"[\W_]+", "", token.casefold())


def merge_chunk_transcript(existing_text: str, new_text: str) -> str:
    existing = existing_text.strip()
    incoming = new_text.strip()
    if not existing:
        return incoming
    if not incoming:
        return existing

    existing_words = existing.split()
    incoming_words = incoming.split()
    existing_norm = [_normalize_token_for_match(word) for word in existing_words]
    incoming_norm = [_normalize_token_for_match(word) for word in incoming_words]

    max_overlap_words = min(len(existing_words), len(incoming_words), 12)
    for overlap_size in range(max_overlap_words, 0, -1):
        if existing_norm[-overlap_size:] == incoming_norm[:overlap_size]:
            remainder = " ".join(incoming_words[overlap_size:]).strip()
            if not remainder:
                return existing
            return f"{existing} {remainder}".strip()

    if incoming and incoming[0] in ",.;:!?)]}":
        return f"{existing}{incoming}"
    return f"{existing} {incoming}".strip()


def transcribe_audio_window(
    *,
    model,
    audio: np.ndarray,
    sample_rate: int,
    beam_size: int,
    vad_filter: bool,
    language: str | None = None,
) -> tuple[str, str, float]:
    prepared_audio, prepared_sample_rate = prepare_audio_samples(audio, sample_rate)
    if prepared_audio.size == 0:
        return "", "unknown", 0.0

    transcribe_kwargs = {
        "beam_size": beam_size,
        "vad_filter": vad_filter,
    }
    if language is not None:
        transcribe_kwargs["language"] = language

    try:
        segments, info = model.transcribe(prepared_audio, **transcribe_kwargs)
    except Exception as exc:
        if vad_filter and "silero_vad_v6.onnx" in str(exc):
            logging.warning("VAD asset missing for chunk, retrying without VAD")
            fallback_kwargs = dict(transcribe_kwargs)
            fallback_kwargs["vad_filter"] = False
            segments, info = model.transcribe(prepared_audio, **fallback_kwargs)
        else:
            raise

    text = "".join(segment.text for segment in segments).strip()
    detected_language = str(getattr(info, "language", "unknown") or "unknown")
    detected_probability = float(getattr(info, "language_probability", 0.0) or 0.0)
    return text, detected_language, detected_probability


def transcribe_audio_chunked(
    *,
    model,
    audio: np.ndarray,
    sample_rate: int,
    chunk_duration_sec: float,
    overlap_duration_sec: float,
    beam_size: int,
    vad_filter: bool,
    language: str | None = None,
    on_status: Callable[[str], None] | None = None,
    on_chunk_done: Callable[[int, int, float, float, int], None] | None = None,
    should_cancel: Callable[[], bool] | None = None,
) -> tuple[str, str, float, int]:
    if audio.size == 0:
        return "", "unknown", 0.0, 0

    chunk_sec = max(0.25, float(chunk_duration_sec))
    chunk_samples = max(1, int(sample_rate * chunk_sec))
    overlap_sec = max(0.0, min(float(overlap_duration_sec), max(0.0, chunk_sec - 0.05)))
    overlap_samples = max(0, int(sample_rate * overlap_sec))
    chunk_ranges = _build_chunk_ranges(audio.shape[0], chunk_samples, overlap_samples)
    total_chunks = len(chunk_ranges)

    merged_text = ""
    lang_score: dict[str, float] = {}

    for idx, (start_idx, end_idx) in enumerate(chunk_ranges, start=1):
        if should_cancel is not None and should_cancel():
            raise TranscriptionCancelledError()

        chunk = audio[start_idx:end_idx]
        if chunk.size == 0:
            continue

        if on_status is not None:
            on_status(f"Transcribing chunk {idx}/{total_chunks}...")

        chunk_text, detected_language, detected_probability = transcribe_audio_window(
            model=model,
            audio=chunk,
            sample_rate=sample_rate,
            beam_size=beam_size,
            vad_filter=vad_filter,
            language=language,
        )
        if chunk_text:
            merged_text = merge_chunk_transcript(merged_text, chunk_text)

        if detected_language:
            lang_score[detected_language] = lang_score.get(detected_language, 0.0) + detected_probability

        if on_chunk_done is not None:
            start_sec = start_idx / float(sample_rate)
            end_sec = end_idx / float(sample_rate)
            on_chunk_done(idx, total_chunks, start_sec, end_sec, len(chunk_text))

        if should_cancel is not None and should_cancel():
            raise TranscriptionCancelledError()

    detected_language = "unknown"
    detected_probability = 0.0
    if lang_score:
        detected_language, detected_probability = max(lang_score.items(), key=lambda item: item[1])

    return merged_text.strip(), detected_language, detected_probability, total_chunks


class LocalSTTCore:
    def _runtime_roots(self) -> list[Path]:
        roots: list[Path] = []
        if getattr(sys, "frozen", False):
            meipass = getattr(sys, "_MEIPASS", None)
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

    def _resolve_model_source(self):
        model_path = Path(self.config.model_path)
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

        return self.config.model_size, False

    def _audio_callback(self, indata: np.ndarray, frames: int, callback_time, status) -> None:
        if status:
            logging.warning("Audio status: %s", status)
        audio_chunk = indata.copy()
        self._append_recorded_audio(audio_chunk)
        if getattr(self, "live_mode_session", None) is not None:
            self.live_mode_session.append_chunk(audio_chunk)

    def _resolve_input_device(self, input_device: str | None):
        if input_device is None or input_device.strip() == "":
            return None
        return _to_int_if_numeric(input_device.strip())

    def _input_devices_list(self) -> list[tuple[int, str]]:
        result: list[tuple[int, str]] = []
        try:
            for idx, dev in enumerate(sd.query_devices()):
                if int(dev.get("max_input_channels", 0)) > 0:
                    raw_name = str(dev.get("name", f"Input {idx}"))
                    hostapi_index = int(dev.get("hostapi", -1))
                    hostapi_name = ""
                    if hostapi_index >= 0:
                        try:
                            hostapi_name = str(sd.query_hostapis(hostapi_index).get("name", "")).strip()
                        except Exception:
                            hostapi_name = ""

                    display_name = _normalize_device_name(raw_name)
                    if hostapi_name:
                        display_name = f"{display_name} [{hostapi_name}]"

                    result.append((idx, display_name))
        except Exception:
            logging.exception("Failed to enumerate input devices")
        return result

    def _find_first_input_device(self):
        for idx, _name in self._input_devices_list():
            return idx
        return None

    def _has_any_input_device(self) -> bool:
        return len(self._input_devices_list()) > 0

    def _get_effective_input_device(self):
        if self.input_device is not None:
            return self.input_device
        try:
            default_in, _ = sd.default.device
            if isinstance(default_in, int) and default_in >= 0:
                return default_in
        except Exception:
            logging.exception("Could not read default audio device")
        return self._find_first_input_device()

    def _iter_input_device_candidates(self):
        seen = set()
        preferred = self._get_effective_input_device()
        if preferred is not None:
            preferred = _to_int_if_numeric(preferred)
            if preferred not in seen:
                seen.add(preferred)
                yield preferred
        for idx, _name in self._input_devices_list():
            if idx not in seen:
                seen.add(idx)
                yield idx

    def _sample_rate_candidates(self, device):
        candidates = [int(self.config.sample_rate)]
        try:
            info = sd.query_devices(device, "input")
            default_sr = int(float(info.get("default_samplerate", 0) or 0))
            if default_sr > 0 and default_sr not in candidates:
                candidates.append(default_sr)
        except Exception:
            logging.exception("Could not query sample rate for device %s", device)

        for sr in [16000, 48000, 44100, 8000]:
            if sr not in candidates:
                candidates.append(sr)
        return candidates

    def _open_input_stream_with_fallback(self):
        last_error = None
        for device in self._iter_input_device_candidates():
            for rate in self._sample_rate_candidates(device):
                try:
                    sd.check_input_settings(
                        device=device,
                        channels=self.config.channels,
                        dtype=self.config.dtype,
                        samplerate=rate,
                    )
                    stream = sd.InputStream(
                        samplerate=rate,
                        channels=self.config.channels,
                        dtype=self.config.dtype,
                        device=device,
                        callback=self._audio_callback,
                    )
                    stream.start()
                    return stream, device, rate
                except Exception as exc:
                    last_error = exc
                    logging.warning("Input probe failed: device=%s samplerate=%s error=%s", device, rate, exc)
        if last_error is not None:
            raise last_error
        raise RuntimeError("No input devices available")

    def _log_audio_input_info(self) -> None:
        try:
            effective = self._get_effective_input_device()
            if effective is None:
                logging.error("No input microphone device found")
                return
            info = sd.query_devices(effective)
            # Make sure it actually has input channels, otherwise it's an output device
            if int(info.get("max_input_channels", 0)) == 0:
                logging.warning(f"Device '{info.get('name', effective)}' has 0 input channels. Falling back to default.")
                self.config.input_device = None
                effective = self._get_effective_input_device()
                info = sd.query_devices(effective) if effective is not None else {}
                
            logging.info("Using microphone: %s", info.get("name", effective))
        except Exception:
            logging.exception("Could not resolve input device")

    def _release_modifiers(self) -> None:
        self.os_adapter.release_modifiers()

    def _drain_audio_queue(self) -> None:
        while not self.audio_queue.empty():
            self.audio_chunks.append(self.audio_queue.get_nowait())

    def _append_recorded_audio(self, audio_chunk: np.ndarray) -> None:
        if audio_chunk.size == 0:
            return
        with self.recording_audio_lock:
            self.recording_audio_bytes.extend(audio_chunk.tobytes())
            self.recording_audio_frame_count += int(audio_chunk.shape[0])
        self.recording_audio_event.set()

    def _get_recorded_audio_frame_count(self) -> int:
        with self.recording_audio_lock:
            return int(self.recording_audio_frame_count)

    def _get_recorded_audio_bytes(self) -> bytes:
        with self.recording_audio_lock:
            return bytes(self.recording_audio_bytes)

    def _get_recorded_audio_slice(self, start_frame: int, end_frame: int) -> np.ndarray:
        dtype = np.dtype(self.config.dtype)
        bytes_per_frame = dtype.itemsize * int(self.config.channels)
        start_byte = max(0, start_frame) * bytes_per_frame
        end_byte = max(start_frame, end_frame) * bytes_per_frame
        with self.recording_audio_lock:
            raw_slice = bytes(self.recording_audio_bytes[start_byte:end_byte])

        if not raw_slice:
            return np.array([], dtype=dtype)

        audio = np.frombuffer(raw_slice, dtype=dtype).copy()
        if self.config.channels > 1:
            audio = audio.reshape(-1, self.config.channels)
        return audio

    def _preferred_transcription_language(self) -> str | None:
        language = getattr(self.config, "transcription_language", "auto")
        return None if language == "auto" else str(language)

    def _transcription_mode(self) -> str:
        raw_mode = str(getattr(self.config, "transcription_mode", "full-file") or "full-file").strip().lower()
        if raw_mode in {"live", "live-overlap", "live_overlap", "stream", "semi-stream", "semi-streaming"}:
            return "live-overlap"
        return "full-file"

    def _uses_live_overlap_mode(self) -> bool:
        return self._transcription_mode() == "live-overlap"

    def _start_live_mode_session(self) -> None:
        self.live_mode_session = live_overlap_mode.LiveOverlapSession(
            model=self.model,
            sample_rate=self.current_recording_sample_rate,
            dtype=self.config.dtype,
            channels=self.config.channels,
            beam_size=self.config.beam_size,
            vad_filter=self.config.vad_filter,
            language=self._preferred_transcription_language(),
            chunk_duration_sec=self.config.chunk_duration_sec,
            chunk_overlap_sec=getattr(self.config, "chunk_overlap_sec", 0.5),
            cancel_event=self.transcription_cancel_event,
            on_status=self._set_status,
        )
        self.is_transcribing = True
        self.live_mode_session.start()

    def _finish_live_mode_session(self, duration_sec: float) -> None:
        session = getattr(self, "live_mode_session", None)
        self.live_mode_session = None
        self.is_transcribing = False

        if session is None:
            self._set_status("Experimental mode session missing")
            self._show_popup("EXPERIMENTAL SESSION MISSING", bg="#9a1b1b")
            self.transcription_cancel_event.clear()
            return

        session.finish_recording()
        session.wait()
        result = session.result

        if result.error is not None:
            self._set_status("Transcription failed")
            self._show_popup("TRANSCRIPTION FAILED", bg="#9a1b1b")
            self.transcription_cancel_event.clear()
            return

        if result.cancelled or self.transcription_cancel_event.is_set():
            self._set_status("Transcription cancelled")
            self._show_popup("TRANSCRIPTION CANCELLED", bg="#7d5a11")
            self.transcription_cancel_event.clear()
            return

        text = result.text.strip()
        logging.info(
            "Live transcription finalized | audio_duration=%.2f sec | chunks=%d | detected language=%s score=%.3f | chars=%d",
            duration_sec,
            result.chunk_count,
            result.detected_language,
            result.detected_score,
            len(text),
        )

        if text:
            remember_result = getattr(self, "_remember_transcription_result", None)
            if callable(remember_result):
                remember_result(text=text, language=result.detected_language, mode="live-overlap")
            self._paste_text(text)
            logging.info("Text pasted to active window")
            self._set_status("Done")
            self._hide_popup()
        else:
            logging.warning("Transcription returned empty text")
            self._set_status("Empty transcription")
            self._show_popup("EMPTY TRANSCRIPTION", bg="#7d5a11")

        self.transcription_cancel_event.clear()

    def _transcribe_full_file(self, audio_path: Path, vad_filter: bool) -> tuple[str, str, float]:
        transcribe_kwargs = {
            "beam_size": self.config.beam_size,
            "vad_filter": vad_filter,
        }
        preferred_language = self._preferred_transcription_language()
        if preferred_language is not None:
            transcribe_kwargs["language"] = preferred_language

        segments, info = self.model.transcribe(str(audio_path), **transcribe_kwargs)
        text_parts: list[str] = []
        for segment in segments:
            if self.transcription_cancel_event.is_set():
                raise live_overlap_mode.TranscriptionCancelledError()
            text_parts.append(segment.text)

        if self.transcription_cancel_event.is_set():
            raise live_overlap_mode.TranscriptionCancelledError()

        return (
            "".join(text_parts).strip(),
            str(getattr(info, "language", "unknown") or "unknown"),
            float(getattr(info, "language_probability", 0.0) or 0.0),
        )

    def _transcribe_selected_file(self, audio_path: Path, vad_filter: bool) -> tuple[str, str, float, int, str]:
        if self._uses_live_overlap_mode():
            text, language, language_score, total_chunks = self._transcribe_chunked(audio_path, vad_filter=vad_filter)
            return text, language, language_score, total_chunks, "live-overlap"

        text, language, language_score = self._transcribe_full_file(audio_path, vad_filter=vad_filter)
        return text, language, language_score, 1, "full-file"

    def _start_live_transcription_worker(self) -> None:
        self.live_transcription_text = ""
        self.live_transcription_chunk_count = 0
        self.live_transcription_language_scores = {}
        self.live_transcription_error = None
        self.live_transcription_cancelled = False
        self.transcription_cancel_event.clear()
        self.is_transcribing = True
        self.live_transcription_thread = threading.Thread(target=self._live_transcription_worker, daemon=True)
        self.live_transcription_thread.start()

    def _join_live_transcription_worker(self) -> None:
        thread = self.live_transcription_thread
        if thread is None:
            return
        self.recording_audio_event.set()
        thread.join()
        self.live_transcription_thread = None

    def _live_transcription_worker(self) -> None:
        chunk_sec = max(0.25, float(self.config.chunk_duration_sec))
        overlap_sec = max(0.0, min(float(getattr(self.config, "chunk_overlap_sec", 0.5)), max(0.0, chunk_sec - 0.05)))
        chunk_samples = max(1, int(round(self.current_recording_sample_rate * chunk_sec)))
        overlap_samples = max(0, int(round(self.current_recording_sample_rate * overlap_sec)))
        step_samples = max(1, chunk_samples - overlap_samples)
        next_chunk_start = 0
        merged_text = ""
        language_scores: dict[str, float] = {}
        processed_chunks = 0
        preferred_language = self._preferred_transcription_language()

        try:
            while True:
                if self.transcription_cancel_event.is_set():
                    raise TranscriptionCancelledError()

                available_frames = self._get_recorded_audio_frame_count()
                full_chunk_ready = available_frames >= next_chunk_start + chunk_samples
                final_tail_ready = (not self.is_recording) and (available_frames > next_chunk_start)

                if not full_chunk_ready and not final_tail_ready:
                    if not self.is_recording:
                        break
                    self.recording_audio_event.wait(timeout=0.1)
                    self.recording_audio_event.clear()
                    continue

                chunk_end = min(available_frames, next_chunk_start + chunk_samples)
                raw_chunk = self._get_recorded_audio_slice(next_chunk_start, chunk_end)
                if raw_chunk.size == 0:
                    if not self.is_recording:
                        break
                    self.recording_audio_event.wait(timeout=0.05)
                    self.recording_audio_event.clear()
                    continue

                processed_chunks += 1
                self._set_status(f"Live transcribing chunk {processed_chunks}...")
                chunk_text, detected_language, detected_probability = transcribe_audio_window(
                    model=self.model,
                    audio=raw_chunk,
                    sample_rate=self.current_recording_sample_rate,
                    beam_size=self.config.beam_size,
                    vad_filter=self.config.vad_filter,
                    language=preferred_language,
                )
                if chunk_text:
                    merged_text = merge_chunk_transcript(merged_text, chunk_text)
                if detected_language:
                    language_scores[detected_language] = language_scores.get(detected_language, 0.0) + detected_probability

                logging.info(
                    "Live chunk %d transcribed | %.2f-%.2f sec | chars=%d | overlap=%.2f sec",
                    processed_chunks,
                    next_chunk_start / float(self.current_recording_sample_rate),
                    chunk_end / float(self.current_recording_sample_rate),
                    len(chunk_text),
                    overlap_sec,
                )

                next_chunk_start += step_samples
                if not self.is_recording and chunk_end >= available_frames:
                    break

            self.live_transcription_text = merged_text.strip()
            self.live_transcription_chunk_count = processed_chunks
            self.live_transcription_language_scores = language_scores
        except TranscriptionCancelledError:
            self.live_transcription_cancelled = True
            logging.info("Live transcription cancelled")
        except Exception as exc:
            self.live_transcription_error = exc
            logging.exception("Live transcription failed")
        finally:
            self.is_transcribing = False
            self.recording_audio_event.set()

    def _get_foreground_window(self) -> int | None:
        return self.os_adapter.get_foreground_window()

    def _get_window_title(self, hwnd: int | None) -> str:
        return self.os_adapter.get_window_title(hwnd)

    def _get_window_class(self, hwnd: int | None) -> str:
        return self.os_adapter.get_window_class(hwnd)

    def _get_window_pid(self, hwnd: int | None) -> int | None:
        return self.os_adapter.get_window_pid(hwnd)

    def _describe_window(self, hwnd: int | None) -> str:
        if hwnd is None:
            return "hwnd=None"
        return (
            f"hwnd={hwnd} pid={self._get_window_pid(hwnd)} "
            f"class='{self._get_window_class(hwnd)}' title='{self._get_window_title(hwnd)}'"
        )

    def _capture_target_window(self, reason: str) -> None:
        hwnd = self._get_foreground_window()
        if hwnd is None:
            logging.info("Target capture skipped (%s): no foreground window", reason)
            return
        if self._get_window_pid(hwnd) == os.getpid():
            logging.info("Target capture skipped (%s): foreground is app process", reason)
            return
        self.target_hwnd = hwnd
        self.target_focus_hwnd = self._get_focused_control(hwnd)
        logging.info(
            "Target window captured (%s): %s | focus_hwnd=%s",
            reason,
            self._describe_window(hwnd),
            self.target_focus_hwnd,
        )

    def _get_focused_control(self, hwnd: int | None) -> int | None:
        return self.os_adapter.get_focused_control(hwnd)

    def _activate_window(self, hwnd: int | None) -> None:
        self.os_adapter.activate_window(hwnd, self.target_focus_hwnd)

    def _generate_audio_path(self) -> Path:
        return self.recordings_dir / f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"

    def _transcribe_chunked(self, audio_path: Path, vad_filter: bool) -> tuple[str, str, float, int]:
        audio, sample_rate = live_overlap_mode.load_audio_for_transcription(audio_path)
        preferred_language = self._preferred_transcription_language()

        return live_overlap_mode.transcribe_audio_chunked(
            model=self.model,
            audio=audio,
            sample_rate=sample_rate,
            chunk_duration_sec=self.config.chunk_duration_sec,
            overlap_duration_sec=getattr(self.config, "chunk_overlap_sec", 0.5),
            beam_size=self.config.beam_size,
            vad_filter=vad_filter,
            language=preferred_language,
            on_status=self._set_status,
            on_chunk_done=lambda idx, total, start_sec, end_sec, chars: logging.info(
                "Chunk %d/%d transcribed | %.2f-%.2f sec | chars=%d",
                idx,
                total,
                start_sec,
                end_sec,
                chars,
            ),
            should_cancel=self.transcription_cancel_event.is_set,
        )

    def cancel_transcription(self) -> None:
        if not self.is_transcribing:
            self._set_status("No transcription is running")
            self._show_popup("NO ACTIVE TRANSCRIPTION", bg="#7d5a11")
            return
        self.transcription_cancel_event.set()
        logging.info("Transcription cancellation requested")
        self._set_status("Cancellation requested...")
        self._show_popup("CANCELLING TRANSCRIPTION...", bg="#7d5a11", persistent=True)

    def repeat_last_paste(self) -> None:
        if not self.last_pasted_text:
            self._set_status("No pasted text to repeat")
            self._show_popup("NOTHING TO RE-PASTE", bg="#7d5a11")
            return
        if self.last_paste_target_hwnd is None:
            self._set_status("Original target is unavailable")
            self._show_popup("PASTE TARGET UNAVAILABLE", bg="#9a1b1b")
            return

        self.target_hwnd = self.last_paste_target_hwnd
        self.target_focus_hwnd = self.last_paste_target_focus_hwnd
        self._paste_text(self.last_pasted_text)
        self._set_status("Last text pasted again")
        self._show_popup("LAST TEXT PASTED AGAIN", bg="#1e6b2d")

    def undo_last_paste(self) -> None:
        if not self.last_paste_can_undo:
            self._set_status("No paste available to undo")
            self._show_popup("NOTHING TO UNDO", bg="#7d5a11")
            return
        if self.last_paste_target_hwnd is None:
            self._set_status("Original target is unavailable")
            self._show_popup("UNDO TARGET UNAVAILABLE", bg="#9a1b1b")
            return

        self.os_adapter.send_undo(self.last_paste_target_hwnd, self.last_paste_target_focus_hwnd)
        self.last_paste_can_undo = False
        logging.info("Undo command sent to last paste target")
        self._set_status("Undo sent to target")
        self._show_popup("UNDO SENT", bg="#1e6b2d")

    def start_recording(self) -> None:
        if self.is_recording:
            return
        if not self._has_any_input_device():
            logging.error("No microphone found")
            self._set_status("Microphone not found")
            self._show_popup("NO MICROPHONE FOUND", bg="#9a1b1b")
            return

        self.audio_chunks.clear()
        self.audio_queue = queue.Queue()
        with self.recording_audio_lock:
            self.recording_audio_bytes = bytearray()
            self.recording_audio_frame_count = 0
        self.recording_audio_event.clear()
        self.live_transcription_text = ""
        self.live_transcription_chunk_count = 0
        self.live_transcription_language_scores = {}
        self.live_transcription_error = None
        self.live_transcription_cancelled = False
        self.live_mode_session = None
        self.is_transcribing = False
        self.transcription_cancel_event.clear()

        try:
            self.stream, chosen_device, chosen_rate = self._open_input_stream_with_fallback()
            self.current_recording_sample_rate = int(chosen_rate)
            self.is_recording = True
            self.recording_started_at = time.perf_counter()
            if self._uses_live_overlap_mode():
                self._start_live_mode_session()
            logging.info("Recording started (device=%s, samplerate=%s)", chosen_device, chosen_rate)
            self._set_status("Recording...")
            self._show_popup("RECORDING...", bg="#9a1b1b", persistent=True)
        except Exception:
            logging.exception("Failed to start recording")
            self._set_status("Microphone open failed")
            self._show_popup("MICROPHONE OPEN FAILED", bg="#9a1b1b")
            self.is_transcribing = False

    def stop_recording(self) -> None:
        if not self.is_recording:
            return
        assert self.stream is not None
        self.stream.stop()
        self.stream.close()
        self.stream = None
        self.is_recording = False
        self.recording_started_at = None

        self._set_status("Recording stopped")
        self._show_popup("RECORDING STOPPED", bg="#1e6b2d", persistent=True)

        audio_bytes = self._get_recorded_audio_bytes()
        total_frames = self._get_recorded_audio_frame_count()
        if not audio_bytes or total_frames <= 0:
            logging.warning("Recording stopped but no audio data was captured")
            self._set_status("No audio captured")
            self._show_popup("NO AUDIO CAPTURED", bg="#7d5a11")
            self.transcription_cancel_event.clear()
            return

        output_path = self._generate_audio_path()
        with wave.open(str(output_path), "wb") as wav_file:
            wav_file.setnchannels(self.config.channels)
            wav_file.setsampwidth(np.dtype(self.config.dtype).itemsize)
            wav_file.setframerate(self.current_recording_sample_rate)
            wav_file.writeframes(audio_bytes)

        self.last_audio_file = output_path
        duration_sec = total_frames / float(self.current_recording_sample_rate)
        logging.info("Recording saved: %s (%.2f sec)", output_path, duration_sec)

        if self._uses_live_overlap_mode():
            self._finish_live_mode_session(duration_sec)
            return

        self._set_status("Transcribing...")
        self._show_popup("TRANSCRIBING FULL FILE...", bg="#0d5f8a", persistent=True)
        self.transcribe_file_async(output_path)

    def toggle_recording(self) -> None:
        try:
            if self.is_recording:
                self.stop_recording()
                return
            self._capture_target_window("toggle_recording_hotkey_start")
            self.start_recording()
        except Exception:
            logging.exception("Toggle recording failed")

    def transcribe_last_file(self) -> None:
        try:
            self._capture_target_window("transcribe_last_hotkey")
            if self.last_audio_file is None:
                logging.warning("No previous recording found")
                self._set_status("No last recording")
                self._show_popup("NO LAST RECORDING", bg="#7d5a11")
                return
            self._set_status("Transcribing...")
            self._show_popup("TRANSCRIBING LAST FILE...", bg="#0d5f8a")
            self.transcribe_file_async(self.last_audio_file)
        except Exception:
            logging.exception("Transcribe last failed")
            self._show_popup("TRANSCRIBE ERROR", bg="#9a1b1b")

    def transcribe_file_async(self, audio_path: Path) -> None:
        if self.is_transcribing:
            logging.warning("Transcription already in progress")
            return
        self.transcription_cancel_event.clear()
        threading.Thread(target=self._transcribe_and_paste, args=(audio_path,), daemon=True).start()

    def _transcribe_and_paste(self, audio_path: Path) -> None:
        self.is_transcribing = True
        start = time.perf_counter()
        try:
            text, language, language_score, total_chunks, mode_name = self._transcribe_selected_file(
                audio_path,
                vad_filter=self.config.vad_filter,
            )
            if self.transcription_cancel_event.is_set():
                raise live_overlap_mode.TranscriptionCancelledError()
            elapsed = time.perf_counter() - start
            if mode_name == "live-overlap":
                logging.info(
                    "Experimental transcription done in %.2f sec | chunks=%d | detected language=%s score=%.3f | chars=%d",
                    elapsed,
                    total_chunks,
                    language,
                    language_score,
                    len(text),
                )
            else:
                logging.info(
                    "Transcription done in %.2f sec | detected language=%s prob=%.3f | chars=%d",
                    elapsed,
                    language,
                    language_score,
                    len(text),
                )
            if text:
                remember_result = getattr(self, "_remember_transcription_result", None)
                if callable(remember_result):
                    remember_result(text=text, language=language, mode=mode_name)
                self._paste_text(text)
                logging.info("Text pasted to active window")
                self._set_status("Done")
                self._hide_popup()
            else:
                logging.warning("Transcription returned empty text")
                self._set_status("Empty transcription")
                self._show_popup("EMPTY TRANSCRIPTION", bg="#7d5a11")
        except live_overlap_mode.TranscriptionCancelledError:
            logging.info("Transcription cancelled")
            self._set_status("Transcription cancelled")
            self._show_popup("TRANSCRIPTION CANCELLED", bg="#7d5a11")
        except Exception as exc:
            if self.config.vad_filter and "silero_vad_v6.onnx" in str(exc):
                try:
                    logging.warning("VAD asset missing, retrying without VAD")
                    text, language, language_score, total_chunks, mode_name = self._transcribe_selected_file(
                        audio_path,
                        vad_filter=False,
                    )
                    if self.transcription_cancel_event.is_set():
                        raise live_overlap_mode.TranscriptionCancelledError()
                    if mode_name == "live-overlap":
                        logging.info(
                            "Experimental transcription done without VAD | chunks=%d | detected language=%s score=%.3f | chars=%d",
                            total_chunks,
                            language,
                            language_score,
                            len(text),
                        )
                    else:
                        logging.info(
                            "Transcription done without VAD | detected language=%s prob=%.3f | chars=%d",
                            language,
                            language_score,
                            len(text),
                        )
                    if text:
                        remember_result = getattr(self, "_remember_transcription_result", None)
                        if callable(remember_result):
                            remember_result(text=text, language=language, mode=mode_name)
                        self._paste_text(text)
                        logging.info("Text pasted to active window")
                        self._set_status("Done (without VAD)")
                        self._hide_popup()
                    else:
                        self._set_status("Empty transcription")
                        self._show_popup("EMPTY TRANSCRIPTION", bg="#7d5a11")
                except live_overlap_mode.TranscriptionCancelledError:
                    logging.info("Transcription cancelled")
                    self._set_status("Transcription cancelled")
                    self._show_popup("TRANSCRIPTION CANCELLED", bg="#7d5a11")
                except Exception:
                    logging.exception("Transcription failed")
                    self._set_status("Transcription failed")
                    self._show_popup("TRANSCRIPTION FAILED", bg="#9a1b1b")
            else:
                logging.exception("Transcription failed")
                self._set_status("Transcription failed")
                self._show_popup("TRANSCRIPTION FAILED", bg="#9a1b1b")
        finally:
            self.is_transcribing = False
            self.transcription_cancel_event.clear()

    def _paste_text(self, text: str) -> None:
        previous_clipboard = None
        try:
            previous_clipboard = pyperclip.paste()
        except Exception:
            logging.warning("Could not read clipboard")

        if self.target_hwnd is None:
            self.target_hwnd = self._get_foreground_window()

        is_app_target = False
        if self.target_hwnd is not None:
            if self._get_window_pid(self.target_hwnd) == os.getpid():
                is_app_target = True

        if is_app_target:
            logging.info("Target window is the application itself. Text copied to clipboard only.")
            pyperclip.copy(text)
            self._set_status("Text copied to clipboard")
            return

        pyperclip.copy(text)
        time.sleep(max(0.25, self.config.paste_delay_sec))

        # Target focus should be captured before starting recording or during _capture_target_window
        self.os_adapter.send_paste(self.target_hwnd)

        self.last_pasted_text = text
        self.last_paste_target_hwnd = self.target_hwnd
        self.last_paste_target_focus_hwnd = self.target_focus_hwnd
        self.last_paste_can_undo = True

        if self.config.restore_clipboard and previous_clipboard is not None:
            try:
                time.sleep(max(0.6, self.config.paste_delay_sec))
                pyperclip.copy(previous_clipboard)
            except Exception:
                logging.warning("Could not restore clipboard")

    def _set_status(self, text: str) -> None:
        logging.info("STATUS: %s", text)

    def _show_popup(self, text: str, bg: str = "#1e6b2d", persistent: bool = False) -> None:
        del text, bg, persistent

    def _hide_popup(self) -> None:
        return None