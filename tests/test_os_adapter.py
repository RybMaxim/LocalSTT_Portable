import pytest
from unittest.mock import MagicMock, patch
from os_adapter import MacAdapter

@pytest.fixture
def mock_keyboard_controller():
    return MagicMock()

@pytest.fixture
def mac_adapter(mock_keyboard_controller):
    with patch('pyautogui.keyUp'): # Mock pyautogui to avoid actual key presses
        return MacAdapter(mock_keyboard_controller)

def test_mac_adapter_get_foreground_window(mac_adapter):
    with patch('subprocess.run') as mock_run:
        mock_run.return_value.stdout = "1234\n"
        pid = mac_adapter.get_foreground_window()
        assert pid == 1234
        mock_run.assert_called_once()
        assert "get unix id of first process whose frontmost is true" in mock_run.call_args[0][0][2]

def test_mac_adapter_get_window_title(mac_adapter):
    with patch('subprocess.run') as mock_run:
        mock_run.return_value.stdout = "Terminal\n"
        title = mac_adapter.get_window_title(1234)
        assert title == "Terminal"
        mock_run.assert_called_once()
        assert "get name of first window of (first process whose unix id is 1234)" in mock_run.call_args[0][0][2]

def test_mac_adapter_activate_window(mac_adapter):
    with patch('subprocess.run') as mock_run:
        mac_adapter.activate_window(1234)
        mock_run.assert_called_once()
        assert "set frontmost of (first process whose unix id is 1234) to true" in mock_run.call_args[0][0][2]

def test_mac_adapter_send_paste(mac_adapter):
    with patch('subprocess.run') as mock_run:
        # Mock activate_window call inside send_paste
        with patch.object(mac_adapter, 'activate_window') as mock_activate:
            success = mac_adapter.send_paste(1234)
            assert success is True
            mock_activate.assert_called_once_with(1234)
            mock_run.assert_called_once()
            assert 'keystroke "v" using {command down}' in mock_run.call_args[0][0][2]

def test_mac_adapter_release_modifiers(mac_adapter, mock_keyboard_controller):
    with patch.object(mac_adapter.pyautogui, 'keyUp') as mock_pyautogui_keyup:
        mac_adapter.release_modifiers()
        # Check if pynput keys are released
        from pynput.keyboard import Key
        mock_keyboard_controller.release.assert_any_call(Key.ctrl)
        mock_keyboard_controller.release.assert_any_call(Key.shift)
        mock_keyboard_controller.release.assert_any_call(Key.alt)
        mock_keyboard_controller.release.assert_any_call(Key.cmd)
        
        # Check if pyautogui keys are released
        mock_pyautogui_keyup.assert_any_call("command")
        mock_pyautogui_keyup.assert_any_call("ctrl")
