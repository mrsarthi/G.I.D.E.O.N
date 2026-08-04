"""
G.I.D.E.O.N Central Configuration & Logging

Single source of truth for models, ports, paths and tuning knobs.
Every value can be overridden with an environment variable so the same
codebase runs on a 4GB laptop GPU or a workstation without edits.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
#  Models
# ---------------------------------------------------------------------------
CHAT_MODEL = _env("GIDEON_CHAT_MODEL", "llama3.2:latest")
EMBEDDING_MODEL = _env("GIDEON_EMBED_MODEL", "nomic-embed-text")

# Context budget. num_ctx is what Ollama allocates KV cache for; the prompt
# budget below is what we are allowed to fill with conversation, leaving room
# for the tool schemas Ollama prepends and for the response itself.
NUM_CTX = _env_int("GIDEON_NUM_CTX", 8192)
TOOL_SCHEMA_RESERVE = _env_int("GIDEON_TOOL_RESERVE", 1400)
RESPONSE_RESERVE = _env_int("GIDEON_RESPONSE_RESERVE", 1024)
MEMORY_RESERVE = _env_int("GIDEON_MEMORY_RESERVE", 700)
CHARS_PER_TOKEN = 4

# What is left over for system prompt + chat history.
PROMPT_TOKEN_BUDGET = max(
    1024, NUM_CTX - TOOL_SCHEMA_RESERVE - RESPONSE_RESERVE - MEMORY_RESERVE
)

# Keep the chat model resident; it is the only thing that should hold VRAM.
CHAT_KEEP_ALIVE = _env("GIDEON_KEEP_ALIVE", "60m")

# The embedder is deliberately pinned to CPU. On a 4GB card it would otherwise
# evict the chat model on every memory write and thrash VRAM.
EMBED_ON_CPU = _env("GIDEON_EMBED_CPU", "1") == "1"
EMBED_KEEP_ALIVE = _env("GIDEON_EMBED_KEEP_ALIVE", "5m")

OLLAMA_OPTIONS = {
    "num_ctx": NUM_CTX,
    "num_gpu": _env_int("GIDEON_NUM_GPU", 99),
}

EMBED_OPTIONS = {"num_gpu": 0} if EMBED_ON_CPU else {}

# ---------------------------------------------------------------------------
#  Memory / retrieval
# ---------------------------------------------------------------------------
MEMORY_RESULTS = _env_int("GIDEON_MEMORY_RESULTS", 3)
# ChromaDB cosine/L2 distance above which a "memory" is treated as noise.
# Without this every turn injects three arbitrary past messages.
MEMORY_MAX_DISTANCE = float(_env("GIDEON_MEMORY_MAX_DISTANCE", "0.75"))
# Messages shorter than this carry no retrievable meaning ("ok", "thanks").
MEMORY_MIN_CHARS = _env_int("GIDEON_MEMORY_MIN_CHARS", 25)
MEMORY_SNIPPET_CHARS = _env_int("GIDEON_MEMORY_SNIPPET_CHARS", 600)

HISTORY_MESSAGE_LIMIT = _env_int("GIDEON_HISTORY_LIMIT", 30)
MAX_TOOL_TURNS = _env_int("GIDEON_MAX_TOOL_TURNS", 4)

# ---------------------------------------------------------------------------
#  Paths
# ---------------------------------------------------------------------------
DB_PATH = BASE_DIR / _env("GIDEON_DB", "gideon.db")
CHROMA_PATH = BASE_DIR / ".chromadb"
FRONTEND_DIR = BASE_DIR / "frontend"
STATIC_DIR = BASE_DIR / "static"
AUDIO_DIR = STATIC_DIR / "audio"
PIPER_MODEL_DIR = BASE_DIR / "model"

# ---------------------------------------------------------------------------
#  Server
# ---------------------------------------------------------------------------
HOST = _env("GIDEON_HOST", "127.0.0.1")
PORT = _env_int("GIDEON_PORT", 7777)

# Explicit origins. "*" plus a directory-wide static mount was serving the
# whole project (including gideon.db) to any origin.
CORS_ORIGINS = [
    o.strip()
    for o in _env(
        "GIDEON_CORS_ORIGINS",
        f"http://localhost:{PORT},http://127.0.0.1:{PORT}",
    ).split(",")
    if o.strip()
]

# ---------------------------------------------------------------------------
#  Voice
# ---------------------------------------------------------------------------
TTS_VOICE_FALLBACK = _env("GIDEON_TTS_VOICE", "en-GB-SoniaNeural")
# edge-tts is a cloud service; off by default to keep the local-first promise.
ALLOW_CLOUD_TTS = _env("GIDEON_ALLOW_CLOUD_TTS", "0") == "1"
# Generated speech older than this is swept at startup.
AUDIO_RETENTION_HOURS = _env_int("GIDEON_AUDIO_RETENTION_HOURS", 6)
MIN_TTS_SENTENCE_CHARS = _env_int("GIDEON_MIN_TTS_CHARS", 12)

# ---------------------------------------------------------------------------
#  Background daemons
# ---------------------------------------------------------------------------
REMINDER_POLL_SECONDS = _env_int("GIDEON_REMINDER_POLL", 30)

# ---------------------------------------------------------------------------
#  Safety
# ---------------------------------------------------------------------------
# Tools that change machine state need explicit user confirmation in the UI
# before the orchestrator will execute them.
CONFIRM_REQUIRED_TOOLS = {"launch_app"}
REQUIRE_CONFIRMATION = _env("GIDEON_REQUIRE_CONFIRMATION", "0") == "1"

# Small models call tools they were never asked to call. Observed: "say thanks
# in one short sentence" triggered set_clipboard, silently overwriting the
# user's clipboard and doubling latency with a wasted round trip. When gating
# is on, the tool schemas are only offered for messages with operational
# intent — which also removes ~1200 tokens of prefill from ordinary chat.
TOOL_GATING = _env("GIDEON_TOOL_GATING", "1") == "1"


# ---------------------------------------------------------------------------
#  Logging
# ---------------------------------------------------------------------------
_LOG_LEVEL = _env("GIDEON_LOG_LEVEL", "INFO").upper()
_configured = False


def setup_logging() -> None:
    """Install a single root handler. Safe to call more than once."""
    global _configured
    if _configured:
        return
    logging.basicConfig(
        level=getattr(logging, _LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(levelname)-7s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    _configured = True


def get_logger(name: str) -> logging.Logger:
    setup_logging()
    return logging.getLogger(name)
