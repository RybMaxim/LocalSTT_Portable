import logging
import queue

import numpy as np
import sounddevice as sd

from app_settings import (
    COMMON_LANGUAGE_OPTIONS,
    MODEL_ORDER,
    MODEL_SIZE_TO_LABEL,
    _language_display_value,
    _normalize_model_size,
    _normalize_transcription_language,
    is_assemblyai_model,
    model_label,
)


class AppUiMixin:
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
            if self._get_effective_input_device() is None:
                self._set_status("No microphone available")
                return

            self.mic_monitor_stream, chosen_device, chosen_rate = self._open_input_stream_with_fallback(
                callback=self._mic_test_callback
            )
            logging.info("Microphone test started (device=%s, samplerate=%s)", chosen_device, chosen_rate)
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

    def _rescan_audio_devices(self) -> None:
        terminate = getattr(sd, "_terminate", None)
        initialize = getattr(sd, "_initialize", None)
        if not callable(terminate) or not callable(initialize):
            return

        try:
            terminate()
            initialize()
        except Exception:
            logging.exception("Failed to reinitialize audio backend during microphone refresh")

    def _refresh_mic_devices(self) -> None:
        if self.ui_mic_combo is None:
            return

        if self.is_recording:
            self._set_status("Stop recording before refreshing microphones")
            return

        if self.mic_monitor_stream is not None:
            self._stop_mic_test()
            if self.ui_mic_test_btn is not None:
                self.ui_mic_test_btn.configure(text="Test microphone")

        self._rescan_audio_devices()

        items = [f"{idx}: {name}" for idx, name in self._input_devices_list()]
        self.ui_mic_combo["values"] = items

        previously_selected = str(self.config.input_device) if self.config.input_device is not None else ""
        selected_item = ""

        selected = str(self.config.input_device) if self.config.input_device is not None else ""
        if selected:
            for item in items:
                if item.startswith(f"{selected}:"):
                    selected_item = item
                    break

        if not selected_item and selected:
            self.config.input_device = None
            self.input_device = None
            self._save_settings_to_file()

        if not selected_item:
            effective = self._get_effective_input_device()
            if effective is not None:
                for item in items:
                    if item.startswith(f"{effective}:"):
                        selected_item = item
                        break

        if not selected_item and items:
            selected_item = items[0]

        if self.ui_mic_var is not None:
            self.ui_mic_var.set(selected_item)

        if selected_item and (not previously_selected or not selected_item.startswith(f"{previously_selected}:")):
            self._apply_selected_mic()
        else:
            self._set_status("Microphone list refreshed")

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
        if keysym in {"c", "СЃ", "cyrillic_es"}:
            return self._copy_text_widget_selection(widget)
        if char in {"c", "СЃ"}:
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
        self._update_model_controls()

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
            background=[("selected", default_bg), ("active", default_bg)],
            foreground=[("selected", "#1f1f1f"), ("active", "#1f1f1f")],
            expand=[("selected", [0, 0, 0, 0])],
        )
        style.configure(
            "TabBody.TFrame",
            background=default_bg,
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
            "Hotkeys (bound to physical A / Q / W / E key positions, independent of layout):\n"
            "Ctrl+Shift+A - cancel current recording and discard audio\n"
            "Ctrl+Shift+Q - start/stop recording\n"
            "Ctrl+Shift+W - transcribe last recording\n"
            "Ctrl+Shift+E - exit\n\n"
            "In Russian layout these are the same physical keys where Р¤ / Р™ / Р¦ / РЈ are printed.\n\n"
            "Default mode transcribes the whole recorded WAV after stop.\n"
            "An experimental live-overlap mode is still available through config if you want to compare it later.\n\n"
            "Set a preferred transcription language in settings to avoid language guessing.\n\n"
            "Use the compact model controls in the Microphone tab to switch between Tiny, Small, Medium, and AssemblyAI.\n"
            "When AssemblyAI is selected, enter your own API key in the Microphone tab.\n"
            "AssemblyAI uses Universal-3 Pro with Universal-2 fallback for unsupported languages.\n\n"
            "How it works:\n"
            "1) Start recording with a hotkey.\n"
            "2) In stable mode, after stop the full WAV is sent either to faster-whisper or to AssemblyAI, depending on the selected provider.\n"
            "3) Recognized text is pasted into the active window.\n\n"
            "The Logs tab shows progress in real time.\n"
            "The History tab keeps the last 10 transcription results."
        )
        desc.configure(state="disabled")
        self._bind_readonly_text_widget(desc)

        mic_scroll_frame = ttk.Frame(tab_mic)
        mic_scroll_frame.pack(fill="both", expand=True)

        mic_canvas = tk.Canvas(
            mic_scroll_frame,
            background=root.cget("bg"),
            highlightthickness=0,
            borderwidth=0,
        )
        mic_scrollbar = ttk.Scrollbar(mic_scroll_frame, orient="vertical", command=mic_canvas.yview)
        mic_canvas.configure(yscrollcommand=mic_scrollbar.set)
        mic_canvas.pack(side="left", fill="both", expand=True)
        mic_scrollbar.pack(side="right", fill="y")

        mic_frame = ttk.Frame(mic_canvas, padding=6, style="TabBody.TFrame")
        mic_window = mic_canvas.create_window((0, 0), window=mic_frame, anchor="nw")

        def _update_mic_scrollregion(_event=None) -> None:
            bbox = mic_canvas.bbox("all")
            if bbox is not None:
                mic_canvas.configure(scrollregion=bbox)

        def _resize_mic_window(event) -> None:
            mic_canvas.itemconfigure(mic_window, width=event.width)

        def _scroll_mic_tab(event):
            step = 0
            delta = getattr(event, "delta", 0)
            if delta:
                step = -1 if delta > 0 else 1
            else:
                button_num = getattr(event, "num", None)
                if button_num == 4:
                    step = -1
                elif button_num == 5:
                    step = 1

            if step != 0:
                mic_canvas.yview_scroll(step, "units")
            return "break"

        mic_frame.bind("<Configure>", _update_mic_scrollregion)
        mic_canvas.bind("<Configure>", _resize_mic_window)

        ttk.Label(mic_frame, text="Input device:").pack(anchor="w")
        self.ui_mic_var = tk.StringVar()
        mic_combo = ttk.Combobox(mic_frame, textvariable=self.ui_mic_var, state="readonly")
        mic_combo.pack(fill="x", pady=(2, 6))
        mic_combo.bind("<<ComboboxSelected>>", lambda _event: self._apply_selected_mic())
        self._disable_combobox_mousewheel(mic_combo, _scroll_mic_tab)
        self.ui_mic_combo = mic_combo

        row = ttk.Frame(mic_frame)
        row.pack(fill="x", pady=(0, 6))
        ttk.Button(row, text="Refresh", command=self._refresh_mic_devices).pack(side="left")

        self.ui_mic_level_var = tk.DoubleVar(value=0.0)
        ttk.Label(mic_frame, text="Input level:").pack(anchor="w", pady=(6, 2))
        ttk.Progressbar(mic_frame, maximum=100.0, variable=self.ui_mic_level_var).pack(fill="x")

        self.ui_mic_test_btn = ttk.Button(mic_frame, text="Test microphone", command=self._toggle_mic_test)
        self.ui_mic_test_btn.pack(fill="x", pady=(8, 6))

        self.ui_vad_var = tk.BooleanVar(value=self.config.vad_filter)
        self.ui_restore_clipboard_var = tk.BooleanVar(value=self.config.restore_clipboard)
        self.ui_language_var = tk.StringVar(value=_language_display_value(self.config.transcription_language))

        ttk.Label(mic_frame, text="Preferred transcription language:").pack(anchor="w", pady=(8, 2))
        language_combo = ttk.Combobox(
            mic_frame,
            textvariable=self.ui_language_var,
            values=[label for label, _code in COMMON_LANGUAGE_OPTIONS],
        )
        language_combo.pack(fill="x")
        self._disable_combobox_mousewheel(language_combo, _scroll_mic_tab)
        self.ui_language_combo = language_combo
        ttk.Label(
            mic_frame,
            text="Use Auto detect or type a Whisper language code such as en, ru, es.",
        ).pack(anchor="w", pady=(2, 6))

        ttk.Label(mic_frame, text="Speech model:").pack(anchor="w", pady=(8, 2))

        self.ui_model_current_var = tk.StringVar(value="")
        ttk.Label(
            mic_frame,
            textvariable=self.ui_model_current_var,
            wraplength=450,
            justify="left",
        ).pack(anchor="w", fill="x", pady=(0, 2))

        model_row = ttk.Frame(mic_frame)
        model_row.pack(fill="x", pady=(0, 2))

        self.ui_model_var = tk.StringVar(value=MODEL_SIZE_TO_LABEL.get(self.config.model_size, model_label(self.config.model_size)))
        model_combo = ttk.Combobox(
            model_row,
            textvariable=self.ui_model_var,
            values=[MODEL_SIZE_TO_LABEL[size] for size in MODEL_ORDER],
            state="readonly",
        )
        model_combo.pack(side="left", fill="x", expand=True)
        model_combo.bind("<<ComboboxSelected>>", lambda _event: self._render_model_ui())
        self._disable_combobox_mousewheel(model_combo, _scroll_mic_tab)
        self.ui_model_combo = model_combo

        action_btn = ttk.Button(
            model_row,
            text="Use model",
            command=lambda: self._dispatch_ui_action("selected_model_action", self._handle_selected_model_action),
        )
        action_btn.pack(side="left", padx=(6, 0))
        self.ui_model_action_btn = action_btn

        self.ui_model_details_var = tk.StringVar(value="")
        ttk.Label(
            mic_frame,
            textvariable=self.ui_model_details_var,
            wraplength=450,
            justify="left",
        ).pack(anchor="w", fill="x", pady=(0, 2))

        self.ui_model_hint_var = tk.StringVar(value="")
        ttk.Label(
            mic_frame,
            textvariable=self.ui_model_hint_var,
            wraplength=450,
            justify="left",
        ).pack(anchor="w", fill="x", pady=(0, 6))

        self.ui_assemblyai_frame = ttk.Frame(mic_frame)
        self.ui_assemblyai_key_var = tk.StringVar(value=getattr(self.config, "assemblyai_api_key", ""))

        ttk.Label(self.ui_assemblyai_frame, text="AssemblyAI API key:").pack(anchor="w")
        self.ui_assemblyai_key_entry = ttk.Entry(
            self.ui_assemblyai_frame,
            textvariable=self.ui_assemblyai_key_var,
            show="*",
        )
        self.ui_assemblyai_key_entry.pack(fill="x", pady=(2, 2))

        self.ui_assemblyai_hint_var = tk.StringVar(value="")
        ttk.Label(
            self.ui_assemblyai_frame,
            textvariable=self.ui_assemblyai_hint_var,
            wraplength=450,
            justify="left",
        ).pack(anchor="w", fill="x")

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
            assemblyai_api_key = self.ui_assemblyai_key_var.get() if self.ui_assemblyai_key_var is not None else ""
            if self.ui_language_var is not None:
                self.ui_language_var.set(_language_display_value(selected_language))
            if self.ui_model_var is not None:
                self.ui_model_var.set(MODEL_SIZE_TO_LABEL.get(selected_model, model_label(selected_model)))
            if is_assemblyai_model(selected_model) and self.ui_assemblyai_key_var is not None:
                self.ui_assemblyai_key_var.set(assemblyai_api_key.strip())
            self._dispatch_ui_action(
                "save_settings",
                lambda: self._apply_user_settings(
                    vad_filter=vad_filter,
                    restore_clipboard=restore_clipboard,
                    transcription_language=selected_language,
                    model_size=selected_model,
                    assemblyai_api_key=assemblyai_api_key,
                ),
            )

        ttk.Checkbutton(mic_frame, text="VAD filter", variable=self.ui_vad_var).pack(anchor="w")
        ttk.Checkbutton(mic_frame, text="Restore clipboard", variable=self.ui_restore_clipboard_var).pack(anchor="w")
        ttk.Button(mic_frame, text="Save settings", command=_apply_simple_settings).pack(fill="x", pady=(6, 0))

        self._bind_mousewheel_scroll(mic_frame, _scroll_mic_tab)

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
        self._render_model_ui()

        if self.startup_model_notice:
            self._set_status(self.startup_model_notice)
            self._show_popup("MODEL FALLBACK TO SMALL", bg="#7d5a11")

        notebook.select(tab_logs)

        def _on_close() -> None:
            self.shutdown()

        root.protocol("WM_DELETE_WINDOW", _on_close)
        self._ui_poll()
        root.mainloop()