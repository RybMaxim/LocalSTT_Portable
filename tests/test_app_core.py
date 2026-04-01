import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from app_core import LocalSTTCore

@pytest.fixture
def mock_config():
    config = MagicMock()
    config.sample_rate = 16000
    config.channels = 1
    config.dtype = "float32"
    config.beam_size = 5
    config.vad_filter = True
    config.paste_delay_sec = 0.1
    config.restore_clipboard = True
    config.transcription_language = "en"
    return config

@pytest.fixture
def core(mock_config):
    core = LocalSTTCore()
    core.config = mock_config
    core.os_adapter = MagicMock()
    core.model = MagicMock()
    core.keyboard_controller = MagicMock()
    core.recording_audio_lock = MagicMock()
    core.recording_audio_event = MagicMock()
    core.transcription_cancel_event = MagicMock()
    core.target_hwnd = None
    core.target_focus_hwnd = None
    core.last_pasted_text = None
    core.last_paste_target_hwnd = None
    core.last_paste_target_focus_hwnd = None
    core.last_paste_can_undo = False
    core.is_transcribing = False
    return core

def test_paste_text_delegates_to_adapter(core):
    with patch('pyperclip.copy') as mock_copy, \
         patch('pyperclip.paste') as mock_paste, \
         patch('time.sleep'):
        mock_paste.return_value = "old_clipboard"
        core.target_hwnd = 1234
        core._paste_text("new_text")
        
        mock_copy.assert_any_call("new_text")
        core.os_adapter.send_paste.assert_called_once_with(1234)
        assert core.last_pasted_text == "new_text"
        assert core.last_paste_target_hwnd == 1234

def test_undo_last_paste_delegates_to_adapter(core):
    core.last_paste_can_undo = True
    core.last_paste_target_hwnd = 1234
    core.last_paste_target_focus_hwnd = 5678
    
    core.undo_last_paste()
    
    core.os_adapter.send_undo.assert_called_once_with(1234, 5678)
    assert core.last_paste_can_undo is False
