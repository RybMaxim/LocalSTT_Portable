import abc
import ctypes
import logging
import os
import subprocess
import time
from typing import Optional

try:
    from ctypes import wintypes
except ImportError:
    # Not on Windows
    wintypes = None

class OSAdapter(abc.ABC):
    @abc.abstractmethod
    def get_foreground_window(self) -> Optional[int]:
        """Returns the handle (or PID) of the current foreground window."""
        pass

    @abc.abstractmethod
    def get_window_title(self, hwnd: Optional[int]) -> str:
        """Returns the title of the window specified by handle."""
        pass

    @abc.abstractmethod
    def get_window_class(self, hwnd: Optional[int]) -> str:
        """Returns the class name of the window specified by handle."""
        pass

    @abc.abstractmethod
    def get_window_pid(self, hwnd: Optional[int]) -> Optional[int]:
        """Returns the process ID of the window specified by handle."""
        pass

    @abc.abstractmethod
    def get_focused_control(self, hwnd: Optional[int]) -> Optional[int]:
        """Returns the handle of the focused control within the given window."""
        pass

    @abc.abstractmethod
    def activate_window(self, hwnd: Optional[int], focus_hwnd: Optional[int] = None) -> None:
        """Brings the specified window to the foreground and sets focus."""
        pass

    @abc.abstractmethod
    def send_paste(self, hwnd: Optional[int]) -> bool:
        """Sends a paste command to the specified window using OS-specific methods."""
        pass

    @abc.abstractmethod
    def send_undo(self, hwnd: Optional[int], focus_hwnd: Optional[int] = None) -> None:
        """Sends an undo command (e.g., Ctrl+Z or Cmd+Z) to the specified window."""
        pass

    @abc.abstractmethod
    def release_modifiers(self) -> None:
        """Releases all modifier keys (Ctrl, Shift, Alt, Cmd/Win)."""
        pass

if wintypes:
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

    class WindowsAdapter(OSAdapter):
        def __init__(self, keyboard_controller):
            self.keyboard_controller = keyboard_controller
            import pyautogui
            self.pyautogui = pyautogui

        def get_foreground_window(self) -> Optional[int]:
            try:
                hwnd = int(ctypes.windll.user32.GetForegroundWindow())
                return hwnd if hwnd != 0 else None
            except Exception:
                return None

        def get_window_title(self, hwnd: Optional[int]) -> str:
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

        def get_window_class(self, hwnd: Optional[int]) -> str:
            if hwnd is None:
                return ""
            try:
                buf = ctypes.create_unicode_buffer(256)
                ctypes.windll.user32.GetClassNameW(hwnd, buf, 255)
                return buf.value
            except Exception:
                return ""

        def get_window_pid(self, hwnd: Optional[int]) -> Optional[int]:
            if hwnd is None:
                return None
            try:
                pid = wintypes.DWORD(0)
                ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                return int(pid.value)
            except Exception:
                return None

        def get_focused_control(self, hwnd: Optional[int]) -> Optional[int]:
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

        def activate_window(self, hwnd: Optional[int], focus_hwnd: Optional[int] = None) -> None:
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
                    if focus_hwnd is not None:
                        user32.SetFocus(focus_hwnd)
                    time.sleep(0.06)
                finally:
                    if attached_current:
                        user32.AttachThreadInput(this_thread, current_thread, False)
                    if attached_target:
                        user32.AttachThreadInput(this_thread, target_thread, False)
            except Exception:
                logging.warning("Could not activate target window")

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

        def send_paste(self, hwnd: Optional[int]) -> bool:
            if hwnd is None:
                return False
            try:
                ctypes.windll.user32.SendMessageW(hwnd, 0x0302, 0, 0)
                time.sleep(0.1)
                self._send_shortcut_vk(0x11, 0x56) # Ctrl+V
                time.sleep(0.1)
                self._send_shortcut_vk(0x10, 0x2D) # Shift+Insert
                return True
            except Exception:
                logging.exception("Paste failed")
                return False

        def send_undo(self, hwnd: Optional[int], focus_hwnd: Optional[int] = None) -> None:
            if hwnd is None:
                return
            self.activate_window(hwnd, focus_hwnd)
            time.sleep(0.1)
            self.release_modifiers()
            time.sleep(0.05)
            self._send_shortcut_vk(0x11, 0x5A) # Ctrl+Z

        def release_modifiers(self) -> None:
            from pynput.keyboard import Key
            for key in [Key.ctrl, Key.shift, Key.alt, Key.cmd]:
                try:
                    self.keyboard_controller.release(key)
                except Exception:
                    pass
            for key_name in ["ctrl", "shift", "alt", "winleft", "winright"]:
                try:
                    self.pyautogui.keyUp(key_name)
                except Exception:
                    pass
else:
    class WindowsAdapter(OSAdapter):
        def __init__(self, keyboard_controller):
            raise RuntimeError("WindowsAdapter requires Windows")
        def get_foreground_window(self) -> Optional[int]: return None
        def get_window_title(self, hwnd: Optional[int]) -> str: return ""
        def get_window_class(self, hwnd: Optional[int]) -> str: return ""
        def get_window_pid(self, hwnd: Optional[int]) -> Optional[int]: return None
        def get_focused_control(self, hwnd: Optional[int]) -> Optional[int]: return None
        def activate_window(self, hwnd: Optional[int], focus_hwnd: Optional[int] = None) -> None: pass
        def send_paste(self, hwnd: Optional[int]) -> bool: return False
        def send_undo(self, hwnd: Optional[int], focus_hwnd: Optional[int] = None) -> None: pass
        def release_modifiers(self) -> None: pass

class MacAdapter(OSAdapter):
    def __init__(self, keyboard_controller):
        self.keyboard_controller = keyboard_controller
        import pyautogui
        self.pyautogui = pyautogui

    def _run_osascript(self, script: str) -> str:
        try:
            result = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, check=True)
            return result.stdout.strip()
        except Exception as e:
            logging.debug(f"osascript error: {e}")
            return ""

    def get_foreground_window(self) -> Optional[int]:
        script = 'tell application "System Events" to get unix id of first process whose frontmost is true'
        pid_str = self._run_osascript(script)
        try:
            return int(pid_str) if pid_str else None
        except ValueError:
            return None

    def get_window_title(self, hwnd: Optional[int]) -> str:
        if hwnd is None: return ""
        script = f'tell application "System Events" to get name of first window of (first process whose unix id is {hwnd})'
        return self._run_osascript(script)

    def get_window_class(self, hwnd: Optional[int]) -> str:
        if hwnd is None: return ""
        script = f'tell application "System Events" to get name of (first process whose unix id is {hwnd})'
        return self._run_osascript(script)

    def get_window_pid(self, hwnd: Optional[int]) -> Optional[int]:
        return hwnd

    def get_focused_control(self, hwnd: Optional[int]) -> Optional[int]:
        return None

    def activate_window(self, hwnd: Optional[int], focus_hwnd: Optional[int] = None) -> None:
        if hwnd is None: return
        script = f'tell application "System Events" to set frontmost of (first process whose unix id is {hwnd}) to true'
        self._run_osascript(script)
        time.sleep(0.1)

    def send_paste(self, hwnd: Optional[int]) -> bool:
        self.activate_window(hwnd)
        script = 'tell application "System Events" to keystroke "v" using {command down}'
        self._run_osascript(script)
        return True

    def send_undo(self, hwnd: Optional[int], focus_hwnd: Optional[int] = None) -> None:
        self.activate_window(hwnd)
        script = 'tell application "System Events" to keystroke "z" using {command down}'
        self._run_osascript(script)

    def release_modifiers(self) -> None:
        from pynput.keyboard import Key
        for key in [Key.ctrl, Key.shift, Key.alt, Key.cmd]:
            try:
                self.keyboard_controller.release(key)
            except Exception:
                pass
        for key_name in ["ctrl", "shift", "alt", "command"]:
            try:
                self.pyautogui.keyUp(key_name)
            except Exception:
                pass

def get_os_adapter(keyboard_controller) -> OSAdapter:
    import sys
    if sys.platform == "win32":
        return WindowsAdapter(keyboard_controller)
    elif sys.platform == "darwin":
        return MacAdapter(keyboard_controller)
    else:
        raise RuntimeError(f"Unsupported platform: {sys.platform}")
