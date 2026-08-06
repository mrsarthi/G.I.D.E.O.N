"""
Unit tests for Kokoro-82M ONNX Speech Synthesis Engine.
"""

from unittest.mock import MagicMock, patch
from pathlib import Path
import os
import pytest

from server import init_kokoro_voice, generate_tts_audio, kokoro_voice
import server


def test_init_kokoro_voice():
    init_kokoro_voice()
    assert server.kokoro_voice is not None, "Kokoro voice should be initialized when model assets exist"


def test_generate_tts_audio_kokoro():
    audio_url = generate_tts_audio("Hello boss, Kokoro synthesis test phrase.", "test_session_123")
    assert audio_url.startswith("/static/audio/")
    assert audio_url.endswith(".wav")

    # Verify physical file creation
    filename = audio_url.replace("/static/audio/", "")
    file_path = server.AUDIO_DIR / filename
    assert file_path.exists()
    assert file_path.stat().st_size > 1000


def test_generate_tts_audio_empty_string():
    audio_url = generate_tts_audio("   ", "test_session_empty")
    assert audio_url == ""
