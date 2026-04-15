import ctypes
import logging
import os
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path

import pyautogui
import pyperclip
from pynput.keyboard import Key

from app_settings import ASSEMBLYAI_SPEECH_MODELS, is_assemblyai_model
from audio_runtime import (
    TranscriptionCancelledError,
    load_audio_for_transcription,
    merge_chunk_transcript,
    transcribe_audio_chunked,
    transcribe_audio_window,
)


class GUITHREADINFO(ctypes.Structure):
    _fields_ = [
        ("cbSize", wintypes.DWORD),
        ("flags", wintypes.DWORD),
        ("hwndActive", wintypes.HWND),
        ("hwndFocus", wintypes.HWND),
        ("hwndCapture", wintypes.HWND),
        ("hwndMenuOwner", wintypes.HWND),
        ("hwndMoveSize", wintypes.HWND),
        ("hwndCaret", wintypes.HWND),
        ("rcCaret", wintypes.RECT),
    ]


@dataclass
class LiveOverlapResult:
    text: str = ""
    detected_language: str = "unknown"
    detected_score: float = 0.0
    chunk_count: int = 0
    cancelled: bool = False
    error: Exception | None = None


class LiveOverlapSession:
    def __init__(
        self,
        *,
        model,
        sample_rate: int,
        dtype: str,
        channels: int,
        beam_size: int,
        vad_filter: bool,
        language: str | None,
        chunk_duration_sec: float,
        chunk_overlap_sec: float,
        cancel_event: threading.Event,
        on_status=None,
    ) -> None:
        self.model = model
        self.sample_rate = int(sample_rate)
        self.dtype = dtype
        self.channels = int(channels)
        self.beam_size = int(beam_size)
        self.vad_filter = bool(vad_filter)
        self.language = language
        self.chunk_duration_sec = float(chunk_duration_sec)
        self.chunk_overlap_sec = float(chunk_overlap_sec)
        self.cancel_event = cancel_event
        self.on_status = on_status

        self.result = LiveOverlapResult()

        self._thread: threading.Thread | None = None
        self._recording_finished = False
        self._audio_lock = threading.Lock()
        self._audio_ready = threading.Event()
        self._audio_bytes = bytearray()
        self._frame_count = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def append_chunk(self, audio_chunk) -> None:
        if audio_chunk.size == 0:
            return
        with self._audio_lock:
            self._audio_bytes.extend(audio_chunk.tobytes())
            self._frame_count += int(audio_chunk.shape[0])
        self._audio_ready.set()

    def finish_recording(self) -> None:
        self._recording_finished = True
        self._audio_ready.set()

    def wait(self) -> None:
        thread = self._thread
        if thread is None:
            return
        thread.join()
        self._thread = None

    def _get_frame_count(self) -> int:
        with self._audio_lock:
            return int(self._frame_count)

    def _get_audio_copy(self):
        with self._audio_lock:
            if not self._audio_bytes:
                return b""
            return bytes(self._audio_bytes)

    def _is_cancelled(self) -> bool:
        return bool(self.cancel_event.is_set())

    def _run(self) -> None:
        try:
            while not self._recording_finished and not self._is_cancelled():
                self._audio_ready.wait(timeout=0.1)
                self._audio_ready.clear()

            if self._is_cancelled():
                self.result.cancelled = True
                return

            audio_bytes = self._get_audio_copy()
            if not audio_bytes:
                self.result.text = ""
                self.result.detected_language = "unknown"
                return

            import numpy as np

            dtype = np.dtype(self.dtype)
            audio = np.frombuffer(audio_bytes, dtype=dtype).copy()
            if self.channels > 1:
                audio = audio.reshape(-1, self.channels)

            text, detected_language, detected_score, chunk_count = transcribe_audio_chunked(
                model=self.model,
                audio=audio,
                sample_rate=self.sample_rate,
                chunk_duration_sec=self.chunk_duration_sec,
                overlap_duration_sec=self.chunk_overlap_sec,
                beam_size=self.beam_size,
                vad_filter=self.vad_filter,
                language=self.language,
                on_status=self.on_status,
                should_cancel=self._is_cancelled,
            )
            self.result.text = text
            self.result.detected_language = detected_language
            self.result.detected_score = detected_score
            self.result.chunk_count = chunk_count
        except TranscriptionCancelledError:
            self.result.cancelled = True
        except Exception as exc:
            self.result.error = exc


class TranscriptionRuntimeMixin:
    def _preferred_transcription_language(self) -> str | None:
        language = getattr(self.config, "transcription_language", "auto")
        return None if language == "auto" else str(language)

    def _transcription_mode(self) -> str:
        raw_mode = str(getattr(self.config, "transcription_mode", "full-file") or "full-file").strip().lower()
        if raw_mode in {"live", "live-overlap", "live_overlap", "stream", "semi-stream", "semi-streaming"}:
            return "live-overlap"
        return "full-file"

    def _uses_live_overlap_mode(self) -> bool:
        if is_assemblyai_model(self.config.model_size):
            return False
        return self._transcription_mode() == "live-overlap"

    def _transcribe_with_assemblyai(self, audio_path: Path) -> tuple[str, str, float]:
        api_key = str(getattr(self.config, "assemblyai_api_key", "") or "").strip()
        if not api_key:
            raise RuntimeError("AssemblyAI API key is missing")

        try:
            import assemblyai as aai
        except ImportError as exc:
            raise RuntimeError("AssemblyAI SDK is not installed") from exc

        aai.settings.base_url = "https://api.assemblyai.com"
        aai.settings.api_key = api_key

        config_kwargs: dict[str, object] = {
            "speech_models": list(ASSEMBLYAI_SPEECH_MODELS),
        }
        preferred_language = self._preferred_transcription_language()
        if preferred_language is None:
            config_kwargs["language_detection"] = True
        else:
            config_kwargs["language_code"] = preferred_language

        self._set_status("Uploading audio to AssemblyAI...")
        transcript = aai.Transcriber().transcribe(
            str(audio_path),
            config=aai.TranscriptionConfig(**config_kwargs),
        )

        if self.transcription_cancel_event.is_set():
            raise TranscriptionCancelledError()

        transcript_error = str(getattr(transcript, "error", "") or "").strip()
        if transcript_error:
            raise RuntimeError(f"AssemblyAI transcription failed: {transcript_error}")

        text = str(getattr(transcript, "text", "") or "").strip()
        language = str(
            getattr(transcript, "language_code", getattr(transcript, "language", "unknown")) or "unknown"
        )
        confidence = float(getattr(transcript, "confidence", 0.0) or 0.0)
        return text, language, confidence

    def _start_live_mode_session(self) -> None:
        self.live_mode_session = LiveOverlapSession(
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
        self._set_transcribing_state(True)
        self.live_mode_session.start()

    def _finish_live_mode_session(self, duration_sec: float) -> None:
        session = getattr(self, "live_mode_session", None)
        self.live_mode_session = None
        self._set_transcribing_state(False)

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
                raise TranscriptionCancelledError()
            text_parts.append(segment.text)

        if self.transcription_cancel_event.is_set():
            raise TranscriptionCancelledError()

        return (
            "".join(text_parts).strip(),
            str(getattr(info, "language", "unknown") or "unknown"),
            float(getattr(info, "language_probability", 0.0) or 0.0),
        )

    def _transcribe_selected_file(self, audio_path: Path, vad_filter: bool) -> tuple[str, str, float, int, str]:
        if is_assemblyai_model(self.config.model_size):
            text, language, language_score = self._transcribe_with_assemblyai(audio_path)
            return text, language, language_score, 1, "assemblyai"

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
        self._set_transcribing_state(True)
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
            self._set_transcribing_state(False)
            self.recording_audio_event.set()

    def _get_foreground_window(self) -> int | None:
        try:
            hwnd = int(ctypes.windll.user32.GetForegroundWindow())
            return hwnd if hwnd != 0 else None
        except Exception:
            return None

    def _get_window_title(self, hwnd: int | None) -> str:
        if hwnd is None:
            return ""
        try:
            user32 = ctypes.windll.user32
            length = int(user32.GetWindowTextLengthW(hwnd))
            if length <= 0:
                return ""
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            return buf.value
        except Exception:
            return ""

    def _get_window_class(self, hwnd: int | None) -> str:
        if hwnd is None:
            return ""
        try:
            buf = ctypes.create_unicode_buffer(256)
            ctypes.windll.user32.GetClassNameW(hwnd, buf, 255)
            return buf.value
        except Exception:
            return ""

    def _get_window_pid(self, hwnd: int | None) -> int | None:
        if hwnd is None:
            return None
        try:
            pid = wintypes.DWORD(0)
            ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            return int(pid.value)
        except Exception:
            return None

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
        if hwnd is None:
            return None
        try:
            user32 = ctypes.windll.user32
            thread_id = int(user32.GetWindowThreadProcessId(hwnd, None))
            if thread_id == 0:
                return None
            gui_info = GUITHREADINFO()
            gui_info.cbSize = ctypes.sizeof(GUITHREADINFO)
            ok = bool(user32.GetGUIThreadInfo(thread_id, ctypes.byref(gui_info)))
            if not ok:
                return None
            focus = int(gui_info.hwndFocus)
            return focus if focus != 0 else None
        except Exception:
            return None

    def _activate_window(self, hwnd: int | None) -> None:
        if hwnd is None:
            return
        try:
            user32 = ctypes.windll.user32
            kernel32 = ctypes.windll.kernel32
            if user32.IsIconic(hwnd):
                user32.ShowWindow(hwnd, 9)

            current_foreground = user32.GetForegroundWindow()
            current_thread = user32.GetWindowThreadProcessId(current_foreground, None)
            target_thread = user32.GetWindowThreadProcessId(hwnd, None)
            this_thread = kernel32.GetCurrentThreadId()

            attached_current = False
            attached_target = False
            try:
                if current_thread and current_thread != this_thread:
                    attached_current = bool(user32.AttachThreadInput(this_thread, current_thread, True))
                if target_thread and target_thread != this_thread:
                    attached_target = bool(user32.AttachThreadInput(this_thread, target_thread, True))

                user32.BringWindowToTop(hwnd)
                user32.SetForegroundWindow(hwnd)
                user32.SetFocus(hwnd)
                if self.target_focus_hwnd is not None:
                    user32.SetFocus(self.target_focus_hwnd)
                time.sleep(0.06)
            finally:
                if attached_current:
                    user32.AttachThreadInput(this_thread, current_thread, False)
                if attached_target:
                    user32.AttachThreadInput(this_thread, target_thread, False)
        except Exception:
            logging.warning("Could not activate target window")

    def _release_modifiers(self) -> None:
        for key in [Key.ctrl, Key.shift, Key.alt, Key.cmd]:
            try:
                self.keyboard_controller.release(key)
            except Exception:
                pass
        for key_name in ["ctrl", "shift", "alt", "winleft", "winright"]:
            try:
                pyautogui.keyUp(key_name)
            except Exception:
                pass

    def _send_vk(self, vk: int, key_up: bool = False) -> None:
        flags = 0x0002 if key_up else 0
        ctypes.windll.user32.keybd_event(vk, 0, flags, 0)

    def _send_shortcut_vk(self, modifier_vk: int, key_vk: int) -> None:
        self._send_vk(modifier_vk, key_up=False)
        time.sleep(0.01)
        self._send_vk(key_vk, key_up=False)
        time.sleep(0.01)
        self._send_vk(key_vk, key_up=True)
        time.sleep(0.01)
        self._send_vk(modifier_vk, key_up=True)

    def _send_wm_paste(self, hwnd: int | None) -> bool:
        if hwnd is None:
            return False
        try:
            ctypes.windll.user32.SendMessageW(hwnd, 0x0302, 0, 0)
            return True
        except Exception:
            logging.exception("WM_PASTE failed")
            return False

    def _generate_audio_path(self) -> Path:
        from datetime import datetime

        return self.recordings_dir / f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"

    def _transcribe_chunked(self, audio_path: Path, vad_filter: bool) -> tuple[str, str, float, int]:
        audio, sample_rate = load_audio_for_transcription(audio_path)
        preferred_language = self._preferred_transcription_language()

        return transcribe_audio_chunked(
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

        self.target_hwnd = self.last_paste_target_hwnd
        self.target_focus_hwnd = self.last_paste_target_focus_hwnd
        self._activate_window(self.last_paste_target_hwnd)
        time.sleep(max(0.2, self.config.paste_delay_sec))
        self._release_modifiers()
        time.sleep(0.05)
        self._send_shortcut_vk(0x11, 0x5A)
        self.last_paste_can_undo = False
        logging.info("Undo shortcut sent to last paste target")
        self._set_status("Undo sent to target")
        self._show_popup("UNDO SENT", bg="#1e6b2d")

    def transcribe_last_file(self) -> None:
        try:
            self._capture_target_window("transcribe_last_hotkey")
            if self.last_audio_file is None:
                logging.warning("No previous recording found")
                self._set_status("No last recording")
                self._show_popup("NO LAST RECORDING", bg="#7d5a11")
                return
            self._set_status("Transcribing...")
            self._show_popup("TRANSCRIBING LAST FILE...", bg="#0d5f8a", persistent=True, timer_mode="transcribing")
            self.transcribe_file_async(self.last_audio_file)
        except Exception:
            logging.exception("Transcribe last failed")
            self._show_popup("TRANSCRIBE ERROR", bg="#9a1b1b")

    def transcribe_file_async(self, audio_path: Path) -> None:
        if self.is_transcribing:
            logging.warning("Transcription already in progress")
            return
        ensure_active_provider_ready = getattr(self, "_ensure_active_provider_ready", None)
        if callable(ensure_active_provider_ready) and not ensure_active_provider_ready():
            return
        self.transcription_cancel_event.clear()
        self._set_transcribing_state(True, audio_duration_sec=self._audio_duration_sec(audio_path))
        threading.Thread(target=self._transcribe_and_paste, args=(audio_path,), daemon=True).start()

    def _transcribe_and_paste(self, audio_path: Path) -> None:
        start = time.perf_counter()
        try:
            text, language, language_score, total_chunks, mode_name = self._transcribe_selected_file(
                audio_path,
                vad_filter=self.config.vad_filter,
            )
            if self.transcription_cancel_event.is_set():
                raise TranscriptionCancelledError()
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
            elif mode_name == "assemblyai":
                logging.info(
                    "AssemblyAI transcription done in %.2f sec | detected language=%s confidence=%.3f | chars=%d",
                    elapsed,
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
        except TranscriptionCancelledError:
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
                        raise TranscriptionCancelledError()
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
                except TranscriptionCancelledError:
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
            self._set_transcribing_state(False)
            self.transcription_cancel_event.clear()

    def _paste_text(self, text: str) -> None:
        previous_clipboard = None
        try:
            previous_clipboard = pyperclip.paste()
        except Exception:
            logging.warning("Could not read clipboard")

        pyperclip.copy(text)
        time.sleep(max(0.25, self.config.paste_delay_sec))

        if self.target_hwnd is None:
            self.target_hwnd = self._get_foreground_window()

        self._activate_window(self.target_hwnd)
        time.sleep(max(0.3, self.config.paste_delay_sec))
        self._release_modifiers()
        time.sleep(0.08)

        self._send_wm_paste(self.target_hwnd)
        time.sleep(max(0.12, self.config.paste_delay_sec))
        self._send_shortcut_vk(0x11, 0x56)
        time.sleep(max(0.12, self.config.paste_delay_sec))
        self._release_modifiers()

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