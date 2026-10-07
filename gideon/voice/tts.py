"""
G.I.D.E.O.N Speech Synthesis

Engines are tried in order: Kokoro (when GIDEON_TTS_ENGINE=kokoro and its
assets are present), then Piper, then — only if GIDEON_ALLOW_CLOUD_TTS=1 —
edge-tts, which sends the text to Microsoft. Both local engines run on CPU
and cost 0 MB of VRAM.

Output is a .wav/.mp3 under static/audio, returned as its served URL.
"""

from __future__ import annotations

import asyncio
import re
import time
import uuid
import wave
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any

from gideon.core.config import (
    ALLOW_CLOUD_TTS,
    AUDIO_DIR,
    AUDIO_RETENTION_HOURS,
    KOKORO_MODEL_PATH,
    KOKORO_VOICE_NAME,
    KOKORO_VOICES_PATH,
    TTS_ENGINE,
    TTS_VOICE_FALLBACK,
    VOICE_MODEL_DIR,
)
from gideon.core.logs import get_logger

log = get_logger("gideon.voice")

AUDIO_URL_PREFIX = "/static/audio/"

_MARKUP = re.compile(r"[*_`#]+")


def clean_for_speech(text: str) -> str:
    return _MARKUP.sub("", text).strip()


class SpeechSynthesizer:
    def __init__(self, audio_dir: Path = AUDIO_DIR) -> None:
        self.audio_dir = audio_dir
        self.kokoro: Any = None
        self.piper: Any = None
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tts_worker")

    # ------------------------------------------------------------------
    #  Engine loading
    # ------------------------------------------------------------------
    def load(self) -> None:
        """Load the local voices and sweep stale audio. Safe to call once at startup."""
        self.load_kokoro()
        self.load_piper()
        self.sweep()

    def load_kokoro(self) -> None:
        """Load the local Kokoro-82M ONNX voice into CPU RAM (0 MB VRAM)."""
        try:
            from kokoro_onnx import Kokoro
        except ImportError:
            log.warning("kokoro-onnx is not installed; skipping Kokoro TTS")
            return

        if not KOKORO_MODEL_PATH.exists() or not KOKORO_VOICES_PATH.exists():
            log.warning(
                "Kokoro assets missing (%s / %s); run scratch/download_kokoro_models.py",
                KOKORO_MODEL_PATH.name,
                KOKORO_VOICES_PATH.name,
            )
            return

        try:
            log.info("Loading Kokoro-82M ONNX model %s", KOKORO_MODEL_PATH.name)
            self.kokoro = Kokoro(str(KOKORO_MODEL_PATH), str(KOKORO_VOICES_PATH))
            log.info("Kokoro-82M TTS ready (CPU, 0 MB VRAM, voice: %s)", KOKORO_VOICE_NAME)
        except Exception as exc:
            log.error("Kokoro TTS initialisation failed: %s", exc)

    def load_piper(self) -> None:
        """Load the local Piper ONNX voice into RAM (no VRAM cost)."""
        try:
            from piper import PiperVoice
        except ImportError:
            log.warning("piper-tts is not installed; local speech disabled")
            return

        # Kokoro's model is also an .onnx; Piper voices are the ones that ship
        # with a sibling .onnx.json config.
        candidates = [
            p for p in sorted(VOICE_MODEL_DIR.glob("*.onnx"))
            if p.with_suffix(".onnx.json").exists()
        ]
        if not candidates:
            log.warning("No Piper .onnx voice (with .onnx.json) found in %s", VOICE_MODEL_DIR)
            return

        model_path = candidates[0]
        try:
            log.info("Loading Piper voice %s", model_path.name)
            self.piper = PiperVoice.load(str(model_path))
            log.info("Piper voice ready (CPU, 0 MB VRAM)")
        except Exception as exc:
            log.error("Piper initialisation failed: %s", exc)

    def sweep(self) -> None:
        """Delete generated speech older than the retention window.

        Without this the directory grows without bound; it had reached 31 MB.
        """
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        cutoff = time.time() - (AUDIO_RETENTION_HOURS * 3600)
        removed = 0
        for path in self.audio_dir.iterdir():
            if not path.is_file():
                continue
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except OSError as exc:
                log.debug("Could not remove %s: %s", path, exc)
        if removed:
            log.info("Swept %d expired audio file(s)", removed)

    @property
    def engines(self) -> dict[str, bool]:
        return {"kokoro": self.kokoro is not None, "piper": self.piper is not None}

    # ------------------------------------------------------------------
    #  Synthesis
    # ------------------------------------------------------------------
    def path_for(self, audio_url: str) -> Path | None:
        """Local file behind a URL returned by synthesize()."""
        if not audio_url:
            return None
        return self.audio_dir / Path(audio_url).name

    def _new_file(self, session_id: str, suffix: str) -> tuple[Path, str]:
        filename = f"{session_id}_{uuid.uuid4().hex[:8]}{suffix}"
        return self.audio_dir / filename, f"{AUDIO_URL_PREFIX}{filename}"

    def synthesize(self, text: str, session_id: str) -> str:
        """Synthesise one utterance and return its served URL ('' on failure)."""
        clean_text = clean_for_speech(text)
        if not clean_text:
            return ""

        self.audio_dir.mkdir(parents=True, exist_ok=True)

        if self.kokoro is not None and TTS_ENGINE == "kokoro":
            try:
                import soundfile as sf

                filepath, url = self._new_file(session_id, ".wav")
                lang = "en-gb" if KOKORO_VOICE_NAME.startswith(("bm_", "bf_")) else "en-us"
                samples, sample_rate = self.kokoro.create(
                    clean_text, voice=KOKORO_VOICE_NAME, speed=1.0, lang=lang
                )
                sf.write(str(filepath), samples, sample_rate)
                return url
            except Exception as exc:
                log.error("Kokoro synthesis failed: %s", exc)

        if self.piper is not None:
            try:
                filepath, url = self._new_file(session_id, ".wav")
                with wave.open(str(filepath), "wb") as wf:
                    first = True
                    for chunk in self.piper.synthesize(clean_text):
                        if first:
                            wf.setnchannels(chunk.sample_channels)
                            wf.setsampwidth(chunk.sample_width)
                            wf.setframerate(chunk.sample_rate)
                            first = False
                        wf.writeframes(chunk.audio_int16_bytes)
                return url
            except Exception as exc:
                log.error("Piper synthesis failed: %s", exc)

        if not ALLOW_CLOUD_TTS:
            return ""

        # Opt-in cloud fallback. Off by default: it sends your text to Microsoft.
        try:
            import edge_tts

            filepath, url = self._new_file(session_id, ".mp3")

            def _worker() -> None:
                loop = asyncio.new_event_loop()
                try:
                    asyncio.set_event_loop(loop)
                    comm = edge_tts.Communicate(clean_text, TTS_VOICE_FALLBACK)
                    loop.run_until_complete(comm.save(str(filepath)))
                finally:
                    loop.close()

            self._executor.submit(_worker).result(timeout=15)
            return url
        except Exception as exc:
            log.error("Cloud TTS fallback failed: %s", exc)
            return ""

    def submit(self, text: str, session_id: str) -> Future:
        """Synthesise on the worker pool, concurrently with generation."""
        return self._executor.submit(self.synthesize, text, session_id)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
