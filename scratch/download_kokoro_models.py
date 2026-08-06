"""
Download Kokoro-82M ONNX model weights and voice embeddings to model/ directory.
"""

import sys
import urllib.request
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

MODEL_FILE = MODEL_DIR / "kokoro-v1.0.onnx"
VOICES_FILE = MODEL_DIR / "voices-v1.0.bin"


def _download_file(url: str, dest_path: Path):
    if dest_path.exists() and dest_path.stat().st_size > 100000:
        print(f"[Kokoro] {dest_path.name} already exists ({dest_path.stat().st_size / (1024*1024):.1f} MB). Skipping download.")
        return

    print(f"[Kokoro] Downloading {dest_path.name} from {url}...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req) as resp, open(dest_path, "wb") as f:
        total_size = int(resp.headers.get("content-length", 0))
        downloaded = 0
        block_size = 1024 * 1024
        while True:
            buffer = resp.read(block_size)
            if not buffer:
                break
            f.write(buffer)
            downloaded += len(buffer)
            if total_size > 0:
                percent = (downloaded / total_size) * 100
                print(f"  Downloaded {downloaded / (1024*1024):.1f} / {total_size / (1024*1024):.1f} MB ({percent:.1f}%)", end="\r", flush=True)
        print()
    print(f"[Kokoro] Saved {dest_path.name} ({dest_path.stat().st_size / (1024*1024):.1f} MB).")


def main():
    _download_file(MODEL_URL, MODEL_FILE)
    _download_file(VOICES_URL, VOICES_FILE)
    print("[Kokoro] All model assets ready.")


if __name__ == "__main__":
    main()
