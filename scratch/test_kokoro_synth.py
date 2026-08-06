"""
Test Kokoro ONNX voice synthesis on CPU.
"""

import sys
import time
from pathlib import Path
import soundfile as sf

from kokoro_onnx import Kokoro

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_FILE = BASE_DIR / "model" / "kokoro-v1.0.onnx"
VOICES_FILE = BASE_DIR / "model" / "voices-v1.0.bin"

print("[Test] Loading Kokoro-82M model on CPU...")
t0 = time.time()
kokoro = Kokoro(str(MODEL_FILE), str(VOICES_FILE))
t_load = time.time() - t0
print(f"[Test] Model loaded in {t_load:.2f}s.")

print("[Test] Available voices:", kokoro.get_voices()[:10])

phrase = "Hello sir, G.I.D.E.O.N speech synthesis core online."
print(f"[Test] Synthesizing phrase: '{phrase}' with voice 'bm_george'...")
t1 = time.time()
samples, sample_rate = kokoro.create(
    phrase,
    voice="bm_george",
    speed=1.0,
    lang="en-gb",
)
t_synth = time.time() - t1
duration = len(samples) / sample_rate
rtf = t_synth / duration

out_wav = BASE_DIR / "static" / "audio" / "test_kokoro.wav"
out_wav.parent.mkdir(parents=True, exist_ok=True)
sf.write(str(out_wav), samples, sample_rate)

print(f"[Test] Generated {duration:.2f}s audio in {t_synth:.2f}s (RTF: {rtf:.2f}). Saved to {out_wav.name}.")
