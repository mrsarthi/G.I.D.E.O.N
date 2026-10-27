"""
G.I.D.E.O.N Central Configuration

Settings only: models, paths, ports and tuning knobs. Logic that used to live
here moved out — logging to core/logs.py, the API token and command denylist
to core/security.py — so importing configuration has no side effects beyond
reading .env.

Every value can be overridden with an environment variable so the same
codebase runs on a 4GB laptop GPU or a workstation without edits.
"""

from __future__ import annotations

import os
from pathlib import Path

# gideon/core/config.py -> project root
BASE_DIR = Path(__file__).resolve().parents[2]


def _load_env_file() -> None:
    env_file = BASE_DIR / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            key = key.strip()
            val = val.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = val


_load_env_file()


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    try:
        return int(os.environ.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_flag(key: str, default: bool) -> bool:
    return _env(key, "1" if default else "0") == "1"


# ---------------------------------------------------------------------------
#  Models
# ---------------------------------------------------------------------------
# qwen3:4b-instruct since 2026-10-10: most accurate on the eval (80% vs
# llama3.2's 78%), never calls tools during plain chat, asks instead of
# guessing most often, ~3x faster to the first word (FINDINGS F-16). It must
# be the *instruct* tag — plain `qwen3:4b` is thinking-only (F-10).
CHAT_MODEL = _env("GIDEON_CHAT_MODEL", "qwen3:4b-instruct")
EMBEDDING_MODEL = _env("GIDEON_EMBED_MODEL", "nomic-embed-text")

# Context budget. num_ctx is what Ollama allocates KV cache for; the prompt
# budget below is what we are allowed to fill with conversation, leaving room
# for the tool schemas Ollama prepends and for the response itself.
NUM_CTX = _env_int("GIDEON_NUM_CTX", 8192)
# Sized for the current 22 tools (~2,690 tokens). brain/prompt.py measures the
# real cost at import and warns if this drifts, so adding tools cannot quietly
# push the assembled prompt past num_ctx.
TOOL_SCHEMA_RESERVE = _env_int("GIDEON_TOOL_RESERVE", 2800)
RESPONSE_RESERVE = _env_int("GIDEON_RESPONSE_RESERVE", 1024)
MEMORY_RESERVE = _env_int("GIDEON_MEMORY_RESERVE", 700)
CHARS_PER_TOKEN = 4

# Keep the chat model resident; it is the only thing that should hold VRAM.
CHAT_KEEP_ALIVE = _env("GIDEON_KEEP_ALIVE", "60m")

# The embedder is deliberately pinned to CPU. On a 4GB card it would otherwise
# evict the chat model on every memory write and thrash VRAM.
EMBED_ON_CPU = _env_flag("GIDEON_EMBED_CPU", True)
EMBED_KEEP_ALIVE = _env("GIDEON_EMBED_KEEP_ALIVE", "5m")

OLLAMA_OPTIONS = {
    "num_ctx": NUM_CTX,
    "num_gpu": _env_int("GIDEON_NUM_GPU", 99),
}

EMBED_OPTIONS = {"num_gpu": 0} if EMBED_ON_CPU else {}

# Where the chat model runs on battery (see brain/llm.py). This laptop powers
# off if the GPU runs the model on battery, so the default keeps it on the CPU:
#   cpu  CPU only (measured ~17.7 tok/s, 0 MiB VRAM)   off  model paused
#   gpu  unchanged — only for machines whose battery can supply the GPU
BATTERY_MODE = _env("GIDEON_ON_BATTERY", "cpu").strip().lower()
if BATTERY_MODE not in ("cpu", "off", "gpu"):
    BATTERY_MODE = "cpu"
# Physical cores: hyperthreads add heat and drain without adding speed here.
BATTERY_THREADS = _env_int("GIDEON_BATTERY_THREADS", max(1, (os.cpu_count() or 2) // 2))
# How often the engine checks for being unplugged.
POWER_POLL_SECONDS = _env_int("GIDEON_POWER_POLL", 5)
# Load the chat model (and prefill the system prompt) when the engine starts,
# so the first question after boot does not wait for a cold load. Skipped on
# battery: there is no point spending the battery on a model nobody asked for.
WARM_UP_MODEL = _env_flag("GIDEON_WARM_UP", True)

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
# Replay earlier tool calls in the history as tool calls (with a short copy
# of each result), not just the sentence that followed. Without them, past
# turns look like plain chat and the model answers the next command in
# prose instead of calling a tool (F-17 fault 2, F-20).
HISTORY_TOOL_CALLS = _env_flag("GIDEON_HISTORY_TOOL_CALLS", True)
HISTORY_TOOL_RESULT_CHARS = _env_int("GIDEON_HISTORY_TOOL_RESULT_CHARS", 240)
# Longest tool output fed back to the model (~1,500 tokens). A 20,000-char
# document dropped into an 8k context pushes the system prompt off the front.
TOOL_RESULT_MAX_CHARS = _env_int("GIDEON_TOOL_RESULT_MAX_CHARS", 6000)
MAX_TOOL_TURNS = _env_int("GIDEON_MAX_TOOL_TURNS", 4)

# ---------------------------------------------------------------------------
#  Paths
# ---------------------------------------------------------------------------
# Runtime state (database, vector store, logs, WebView profile) lives under
# DATA_DIR instead of littering the project root. Override with
# GIDEON_DATA_DIR — the test suite does, so it never touches real data.
_DATA_DIR_OVERRIDE = _env("GIDEON_DATA_DIR", "")
DATA_DIR = Path(_DATA_DIR_OVERRIDE) if _DATA_DIR_OVERRIDE else BASE_DIR / "data"


def _state_path(name: str, legacy_name: str | None = None) -> Path:
    """Where a piece of runtime state lives.

    Installs that predate DATA_DIR keep their files in the project root.
    Those are used in place until scripts/migrate_data_dir.py moves them —
    never moved implicitly, because a rename that half-succeeds while the
    engine holds the database open would look exactly like data loss.
    """
    preferred = DATA_DIR / name
    if _DATA_DIR_OVERRIDE or preferred.exists():
        return preferred
    legacy = BASE_DIR / (legacy_name or name)
    return legacy if legacy.exists() else preferred


DB_PATH = _state_path(_env("GIDEON_DB", "gideon.db"))
CHROMA_PATH = _state_path("chroma", legacy_name=".chromadb")
LOG_DIR = _state_path("logs")
WEBVIEW_DIR = _state_path("webview", legacy_name=".webview")

# Folders the file-search tool may walk, separated by ';'. Empty means
# Desktop, Documents and Downloads (and their OneDrive-backed versions).
SEARCH_ROOTS = [p for p in _env("GIDEON_SEARCH_ROOTS", "").split(os.pathsep) if p.strip()]

# Project knowledge (storage/projects.py): your code folders, found under the
# search roots and in VS Code's history, with their git state, kept in fact
# memory. Refreshed at start and then every PROJECT_REFRESH_MINUTES.
PROJECT_INDEX = _env_flag("GIDEON_PROJECT_INDEX", True)
PROJECT_REFRESH_MINUTES = _env_int("GIDEON_PROJECT_REFRESH_MINUTES", 30)
PROJECT_SCAN_DEPTH = _env_int("GIDEON_PROJECT_SCAN_DEPTH", 4)

FRONTEND_DIR = BASE_DIR / "frontend"
STATIC_DIR = BASE_DIR / "static"
AUDIO_DIR = STATIC_DIR / "audio"
VOICE_MODEL_DIR = BASE_DIR / "model"

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

# Only these Host header values are served. Blocks DNS rebinding, where a
# hostile page resolves its own domain to 127.0.0.1 to reach this server.
ALLOWED_HOSTS = {
    h.strip().lower()
    for h in _env(
        "GIDEON_ALLOWED_HOSTS", f"localhost:{PORT},127.0.0.1:{PORT},localhost,127.0.0.1"
    ).split(",")
    if h.strip()
}

LOG_LEVEL = _env("GIDEON_LOG_LEVEL", "INFO").upper()

# ---------------------------------------------------------------------------
#  Voice
# ---------------------------------------------------------------------------
# Engine order: Kokoro (if selected and installed) -> Piper -> edge-tts (only
# when ALLOW_CLOUD_TTS). Both local engines run on CPU, 0 MB VRAM.
TTS_ENGINE = _env("GIDEON_TTS_ENGINE", "kokoro")
KOKORO_MODEL_PATH = VOICE_MODEL_DIR / _env("GIDEON_KOKORO_MODEL", "kokoro-v1.0.onnx")
KOKORO_VOICES_PATH = VOICE_MODEL_DIR / _env("GIDEON_KOKORO_VOICES", "voices-v1.0.bin")
KOKORO_VOICE_NAME = _env("GIDEON_KOKORO_VOICE", "bf_emma")
TTS_VOICE_FALLBACK = _env("GIDEON_TTS_VOICE", "en-GB-SoniaNeural")
# edge-tts is a cloud service; off by default to keep the local-first promise.
ALLOW_CLOUD_TTS = _env_flag("GIDEON_ALLOW_CLOUD_TTS", False)
# Generated speech older than this is swept at startup.
AUDIO_RETENTION_HOURS = _env_int("GIDEON_AUDIO_RETENTION_HOURS", 6)
MIN_TTS_SENTENCE_CHARS = _env_int("GIDEON_MIN_TTS_CHARS", 12)

# ---------------------------------------------------------------------------
#  Reminders
# ---------------------------------------------------------------------------
REMINDER_POLL_SECONDS = _env_int("GIDEON_REMINDER_POLL", 30)

# How late a reminder may be and still be delivered as itself. Anything older
# came due while this machine was asleep or shut down; firing those normally
# means a weekend away turns into a stampede of stale alerts with speech
# synthesised for each. Past the window they are retired as 'missed' and
# summarised in one digest instead.
REMINDER_MAX_LATENESS_MINUTES = _env_int("GIDEON_REMINDER_MAX_LATENESS", 15)
# Missed reminders older than this are dropped from the digest entirely; being
# told about last month's is noise, not information.
REMINDER_DIGEST_MAX_AGE_HOURS = _env_int("GIDEON_REMINDER_DIGEST_MAX_AGE", 72)

# ---------------------------------------------------------------------------
#  Remote scheduler (always-on reminder delivery)
# ---------------------------------------------------------------------------
# A reminder's whole value is arriving at a moment this machine does not
# control. When a node URL is set, new reminders are mirrored to it and it
# owns Telegram delivery — so they still fire when the desktop is off. Only
# the reminder text, its due time and the origin chat_id leave the machine;
# conversations, memory and documents never do.
REMOTE_SCHEDULER_URL = _env("GIDEON_REMOTE_SCHEDULER_URL", "").rstrip("/")
REMOTE_SCHEDULER_TOKEN = _env("GIDEON_REMOTE_SCHEDULER_TOKEN", "")
REMOTE_SCHEDULER_TIMEOUT = _env_int("GIDEON_REMOTE_SCHEDULER_TIMEOUT", 8)
# How often unsynced rows are retried. The node being briefly unreachable must
# not cost a reminder, so failures stay queued rather than being dropped.
REMOTE_SCHEDULER_RETRY_SECONDS = _env_int("GIDEON_REMOTE_SCHEDULER_RETRY", 300)

# ---------------------------------------------------------------------------
#  Tray, hotkey and quick-ask window
# ---------------------------------------------------------------------------
# Modifier names are resolved against Win32 MOD_* in interfaces/tray.py.
# Ctrl+Alt+G is unclaimed by Windows; avoid Win+G (Xbox Game Bar) and
# Alt+Space (system menu).
HOTKEY_MODIFIERS = _env("GIDEON_HOTKEY_MODIFIERS", "ctrl+alt")
HOTKEY_KEY = _env("GIDEON_HOTKEY_KEY", "G").upper()

QUICK_WINDOW_WIDTH = _env_int("GIDEON_QUICK_WIDTH", 720)
QUICK_WINDOW_HEIGHT = _env_int("GIDEON_QUICK_HEIGHT", 420)
# Fixed session so quick asks keep their own short thread instead of polluting
# the HUD transcript. Vector memory is shared, so context still carries.
QUICK_SESSION_ID = _env("GIDEON_QUICK_SESSION", "quick-ask")

# Seconds the tray waits for the engine to answer /api/health before deciding
# it has to start one itself. Cold start loads Chroma, Kokoro and Piper.
ENGINE_START_TIMEOUT = _env_int("GIDEON_ENGINE_START_TIMEOUT", 90)

# ---------------------------------------------------------------------------
#  Telegram & external APIs
# ---------------------------------------------------------------------------
TELEGRAM_BOT_TOKEN = _env("GIDEON_TELEGRAM_BOT_TOKEN", "")
TELEGRAM_ALLOWED_USERS = {
    int(u.strip())
    for u in _env("GIDEON_TELEGRAM_ALLOWED_USERS", "").split(",")
    if u.strip().isdigit()
}
TELEGRAM_SEND_VOICE = _env_flag("GIDEON_TELEGRAM_SEND_VOICE", False)

WOLFRAM_ALPHA_APP_ID = _env("GIDEON_WOLFRAM_APP_ID", "")

# ---------------------------------------------------------------------------
#  Safety switches
# ---------------------------------------------------------------------------
# Tools declared with confirm=True (the shell) wait for the user to approve
# them in the HUD or quick window before they run. On by default: a web page
# the model just read must not be able to reach a shell unattended. Tools
# declared privileged=True (shell, app launching) need approval regardless of
# this setting once outside content has entered the turn.
REQUIRE_CONFIRMATION = _env_flag("GIDEON_REQUIRE_CONFIRMATION", True)
# Unanswered approval requests lapse after this long.
APPROVAL_TTL_MINUTES = _env_int("GIDEON_APPROVAL_TTL_MINUTES", 15)

# Small models call tools they were never asked to call. Observed: "say thanks
# in one short sentence" triggered set_clipboard, silently overwriting the
# user's clipboard and doubling latency with a wasted round trip. When gating
# is on, the tool schemas are only offered for messages with operational
# intent — which also removes ~1200 tokens of prefill from ordinary chat.
TOOL_GATING = _env_flag("GIDEON_TOOL_GATING", True)

# Offer only the most relevant tools per message (brain/router.py):
#   auto  on for models whose template re-reads the tool list every turn
#         (llama3.2: 3x faster, same accuracy), off for models that cache it
#         (Qwen: routing breaks the cache and costs accuracy) — FINDINGS F-16
#   on / off  force it. "1"/"0" are accepted as on/off.
_ROUTER = _env("GIDEON_TOOL_ROUTER", "auto").strip().lower()
TOOL_ROUTER = {"1": "on", "true": "on", "0": "off", "false": "off"}.get(_ROUTER, _ROUTER)
if TOOL_ROUTER not in ("auto", "on", "off"):
    TOOL_ROUTER = "auto"

# Which tools may do what (PRD §5.6, §9.2): capability class, confirmation
# and channel rules, editable without touching code.
POLICY_FILE = Path(_env("GIDEON_POLICY_FILE", str(BASE_DIR / "policy.toml")))
ROUTER_TOP_K = _env_int("GIDEON_ROUTER_TOP_K", 5)

ENFORCE_COMMAND_DENYLIST = _env_flag("GIDEON_COMMAND_DENYLIST", True)
TERMINAL_TIMEOUT_SECONDS = _env_int("GIDEON_TERMINAL_TIMEOUT", 30)

