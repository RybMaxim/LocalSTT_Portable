import logging
import queue
import re
import time
import wave
from collections.abc import Callable
from pathlib import Path

import numpy as np
import sounddevice as sd


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
    prepared_audio, _prepared_sample_rate = prepare_audio_samples(audio, sample_rate)
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


class AudioRuntimeMixin:
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

    def _open_input_stream_with_fallback(self, callback=None):
        stream_callback = self._audio_callback if callback is None else callback
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
                        callback=stream_callback,
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
            info = sd.query_devices(effective, "input")
            logging.info("Using microphone: %s", info.get("name", effective))
        except Exception:
            logging.exception("Could not resolve input device")

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

    def _reset_current_recording_buffers(self) -> None:
        self.audio_chunks.clear()
        self.audio_queue = queue.Queue()
        with self.recording_audio_lock:
            self.recording_audio_bytes = bytearray()
            self.recording_audio_frame_count = 0
        self.recording_audio_event.clear()

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

    def _audio_duration_sec(self, audio_path: Path) -> float | None:
        try:
            with wave.open(str(audio_path), "rb") as wav_file:
                frame_count = int(wav_file.getnframes())
                sample_rate = int(wav_file.getframerate())
            if frame_count <= 0 or sample_rate <= 0:
                return None
            return frame_count / float(sample_rate)
        except Exception:
            logging.exception("Failed to read audio duration from: %s", audio_path)
            return None

    def _set_transcribing_state(self, active: bool, audio_duration_sec: float | None = None) -> None:
        self.is_transcribing = active
        if active:
            self.transcription_started_at = time.perf_counter()
            self.transcription_target_duration_sec = audio_duration_sec
        else:
            self.transcription_started_at = None
            self.transcription_target_duration_sec = None

    def start_recording(self) -> None:
        if self.is_recording:
            return
        ensure_active_provider_ready = getattr(self, "_ensure_active_provider_ready", None)
        if callable(ensure_active_provider_ready) and not ensure_active_provider_ready():
            return
        if not self._has_any_input_device():
            logging.error("No microphone found")
            self._set_status("Microphone not found")
            self._show_popup("NO MICROPHONE FOUND", bg="#9a1b1b")
            return

        self._reset_current_recording_buffers()
        self.live_transcription_text = ""
        self.live_transcription_chunk_count = 0
        self.live_transcription_language_scores = {}
        self.live_transcription_error = None
        self.live_transcription_cancelled = False
        self.live_mode_session = None
        self._set_transcribing_state(False)
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
            self._show_popup("RECORDING...", bg="#9a1b1b", persistent=True, timer_mode="recording")
        except Exception:
            logging.exception("Failed to start recording")
            self._set_status("Microphone open failed")
            self._show_popup("MICROPHONE OPEN FAILED", bg="#9a1b1b")
            self._set_transcribing_state(False)

    def cancel_recording(self) -> None:
        if not self.is_recording:
            return

        session = getattr(self, "live_mode_session", None)

        try:
            if self.stream is not None:
                self.stream.stop()
                self.stream.close()
        except Exception:
            logging.exception("Failed to stop stream while cancelling recording")
        finally:
            self.stream = None

        self.is_recording = False
        self.recording_started_at = None

        if session is not None:
            self.transcription_cancel_event.set()
            try:
                session.finish_recording()
                session.wait()
            except Exception:
                logging.exception("Failed to stop live transcription session during recording cancel")

        self.live_mode_session = None
        self.live_transcription_text = ""
        self.live_transcription_chunk_count = 0
        self.live_transcription_language_scores = {}
        self.live_transcription_error = None
        self.live_transcription_cancelled = False
        self._set_transcribing_state(False)
        self.transcription_cancel_event.clear()
        self._reset_current_recording_buffers()

        logging.info("Recording cancelled and discarded")
        self._set_status("Recording cancelled")
        self._show_popup("RECORDING CANCELLED", bg="#7d5a11")

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
        self._show_popup("TRANSCRIBING FULL FILE...", bg="#0d5f8a", persistent=True, timer_mode="transcribing")
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