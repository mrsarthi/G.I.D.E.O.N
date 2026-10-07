"""
Unit tests for Kokoro-82M ONNX Speech Synthesis Engine.

These synthesise real audio, so they need the Kokoro assets in model/.
"""

import pytest

from gideon.core.config import KOKORO_MODEL_PATH, KOKORO_VOICES_PATH
from gideon.voice.tts import AUDIO_URL_PREFIX, SpeechSynthesizer

requires_kokoro = pytest.mark.skipif(
    not (KOKORO_MODEL_PATH.exists() and KOKORO_VOICES_PATH.exists()),
    reason="Kokoro model assets are not downloaded",
)


@pytest.fixture(scope="module")
def synth():
    speech = SpeechSynthesizer()
    speech.load_kokoro()
    yield speech
    speech.shutdown()


@requires_kokoro
def test_init_kokoro_voice(synth):
    assert synth.kokoro is not None, "Kokoro voice should be initialized when model assets exist"


@requires_kokoro
def test_generate_tts_audio_kokoro(synth):
    audio_url = synth.synthesize("Hello boss, Kokoro synthesis test phrase.", "test_session_123")
    assert audio_url.startswith(AUDIO_URL_PREFIX)
    assert audio_url.endswith(".wav")

    # Verify physical file creation
    file_path = synth.path_for(audio_url)
    assert file_path.exists()
    assert file_path.stat().st_size > 1000
    file_path.unlink()


def test_generate_tts_audio_empty_string():
    assert SpeechSynthesizer().synthesize("   ", "test_session_empty") == ""
