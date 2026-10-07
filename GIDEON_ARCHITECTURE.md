# G.I.D.E.O.N — System Architecture & Technical Specification

> **G.I.D.E.O.N** (*Guided Intelligence Does Everything One Needs*) is a privacy-first, local-first AI assistant built on **Ollama**, **FastAPI**, **SQLite**, **ChromaDB**, **Kokoro / Piper TTS**, and **Three.js WebGL**.

---

## 1. High-Level Architecture

```
   ┌──────────────────────┐  ┌────────────┐  ┌──────────┐  ┌───────────┐
   │ 3D HUD · quick ask   │  │  Telegram  │  │   Tray   │  │    CLI    │
   └──────────┬───────────┘  └─────┬──────┘  └────┬─────┘  └─────┬─────┘
              │ SSE /api/chat       │              │ /api/health  │
              ▼ SSE /api/events     │              ▼              │
   ┌──────────────────────┐         │       (supervises engine)   │
   │ interfaces/api       │         │                             │
   │ FastAPI routes       │         │                             │
   └──────────┬───────────┘         │                             │
              ▼                     ▼                             ▼
   ┌────────────────────────────────────────────────────────────────────┐
   │ app.py — composition root: builds every service once, wires them   │
   └──┬──────────────────┬───────────────────┬─────────────────────┬────┘
      ▼                  ▼                   ▼                     ▼
 ┌──────────────┐ ┌──────────────┐ ┌──────────────────┐ ┌────────────────┐
 │ brain/agent  │ │ voice/       │ │ reminders/       │ │ core/events    │
 │ prompt·intent│ │ Kokoro/Piper │ │ delivery·digest  │ │ (event bus)    │
 │ tool loop    │ │ per-sentence │ │ node sync        │ │                │
 └──┬────────┬──┘ └──────────────┘ └────────┬─────────┘ └────────────────┘
    │        ▼                              │
    │  ┌───────────────────┐                │
    │  │ Ollama · llama3.2 │                │
    │  └───────────────────┘                │
    ▼                                       ▼
 ┌──────────────────────┐   ┌─────────────────────────────────────────────┐
 │ tools/ (@tool decl.) │──▶│ storage/  SQLite · ChromaDB · remote mirror │
 └──────────────────────┘   └─────────────────────────────────────────────┘
```

---

## 2. Core Subsystems

### A. Inference & Prompt Assembly
- **Model**: configurable via `GIDEON_CHAT_MODEL` (default `llama3.2:latest`), served by local Ollama.
- **Prefix cache discipline**: message order is
  `[static system] [older history] [live context] [newest user turn]`.
  Index 0 is byte-stable for the life of the process — nothing volatile (clock,
  profile, recalled memories) is allowed near the front, because Ollama only
  reuses cached prefill while the *leading* tokens are identical.
- **Token budget** (`brain/prompt.py`): `num_ctx` minus the tool schemas
  Ollama prepends (measured at import, ~2,240 tokens for 19 tools), minus the
  memory reserve, minus room for the answer. Overflowing this silently drops
  the system prompt off the front.
- **Tool gating** (`brain/intent.py`): tool schemas are only offered for
  messages with operational intent. Small models otherwise fire tools during
  small talk — observed: *"say thanks in one short sentence"* invoked
  `set_clipboard`. The vocabulary is the union of each tool's declared
  `triggers`, so a new tool brings its own.
- **Pseudo tool-call interception** (`brain/toolcalls.py`): replies that open with a JSON object are
  withheld until they can be classified. Models frequently type
  `{"name": ..., "parameters": ...}` as prose; unintercepted it reaches the UI
  and gets read aloud.

### B. Dual-Tier Memory
1. **Relational (SQLite — `storage/db.py`)** — sessions, messages, profile, reminders.
   One shared connection behind an `RLock`, WAL mode, `busy_timeout`, and a
   checkpoint on close. Use `get_db()`; constructing `DatabaseManager` per call
   site opens redundant connections to the same file. Reminders come back as
   typed `Reminder` rows (`storage/models.py`).
2. **Semantic (ChromaDB — `storage/memory.py`, `VectorMemory`)** — embeddings via `nomic-embed-text`,
   **pinned to CPU**. On a 4GB card a GPU embedder evicts the resident chat
   model on every write. Writes are filtered (filler and tool output are never
   indexed) and reads are distance-thresholded, so unrelated memories stop
   being injected into every turn.

### C. Tools (`tools/`)
Native Ollama JSON function calling. Each tool is declared once, beside its
implementation, with `@tool` (`tools/base.py`): description, parameters
(with the aliases small models invent), `blocked_channels`, `confirm`, and
intent `triggers`. `tools/registry.py` derives the schemas, dispatch and
channel filtering from those declarations, and pins the order schemas are
offered in — small models are order-biased, so `play_youtube_music` must
precede `control_media`.

| Tool | Purpose |
| :--- | :--- |
| `get_system_status` | CPU/GPU/RAM/disk/battery telemetry |
| `set_reminder` / `list_reminders` | Scheduling, fired by the reminder service |
| `launch_app` | **Allowlist-only** application launcher (desktop only, confirm) |
| `search_files` | Filename search across approved roots |
| `search_web` | Headless DDG search with Instant-Answer/Wikipedia fallbacks |
| `get_clipboard` / `set_clipboard` | Windows clipboard access |
| `control_volume` | Master volume via pycaw |
| `control_media` | Play/pause/next/previous via media keys |
| `play_youtube_music` / `search_youtube_music` | YouTube Music |
| `read_document` | PDF and text/code extraction |
| `get_weather` | Open-Meteo conditions and forecast |
| `get_financial_data` | Quotes, news and financials via yfinance |
| `get_hackernews` | Hacker News stories |
| `get_ip_info` | IP geolocation |
| `query_wolfram_alpha` | Computation and facts |
| `run_terminal_command` | Shell, behind a denylist and audit trail (desktop only, confirm) |

`launch_app` accepts only names present in `KNOWN_APPS`. The previous
fall-through to `subprocess.Popen(f"start {name}", shell=True)` was a shell
injection with the model holding the pen.

### D. Voice (`voice/`, local)
- **Primary**: Kokoro-82M ONNX (`GIDEON_TTS_ENGINE=kokoro`, the default) on
  CPU — 0 MB VRAM. Falls back to **Piper** ONNX, also CPU.
- **Cloud fallback**: `edge-tts`, **off by default** (`GIDEON_ALLOW_CLOUD_TTS=1` to
  enable) because it sends text to Microsoft.
- **Sentence streaming**: speech is synthesised per completed sentence while
  the model is still generating, and emitted in order. Measured: first audio at
  8.96s on a reply that finished at 12.4s.
- Generated audio is swept at startup (`GIDEON_AUDIO_RETENTION_HOURS`).

### E. Event Bus (`core/events.py`)
A fan-out pub/sub over asyncio queues backing `GET /api/events`. The reminder
service (`reminders/service.py`) polls for due rows, marks them completed, synthesises speech, and
publishes — this is what makes `set_reminder` more than a database write.
Saturated subscribers drop events rather than stalling publishers.

### F. Reminder delivery
Two problems that the naive "poll SQLite and fire" loop got wrong:

**Lateness.** The daemon used to select every pending row with
`remind_at <= now`, so booting after a weekend fired the whole backlog at once
with speech synthesised per reminder. Anything more than
`GIDEON_REMINDER_MAX_LATENESS` (15 min) overdue is now retired as `missed` and
reported as **one digest with one utterance**, not a stampede.

**Uptime.** A reminder's value is arriving at a moment this machine does not
control, so `node/` is a Cloudflare Worker that holds upcoming reminders and
sends them on Telegram — including while the desktop is shut down. Delivery
ownership is exclusive and per row, tracked by `reminders.remote_synced`:

| `remote_synced` | Telegram copy sent by |
| :--- | :--- |
| `1` — node accepted it | **the node** (fires with the PC off) |
| `0` — push failed, or no node configured | **the desktop** |

So nothing is delivered twice, and a node outage degrades to local-only
delivery rather than silence. Only reminder text, due time and `chat_id` leave
the machine — never conversations, memory or documents. The node **only
sends**; it never polls, because Telegram permits one `getUpdates` consumer per
token and that consumer is `interfaces/telegram.py`.

### G. Tray, hotkey and quick ask (`interfaces/tray.py`)
The entry point — you should never have to run a command to reach
G.I.D.E.O.N. It supervises the engine, sits in the notification area, and
answers **Ctrl+Alt+G** from anywhere with a small ask-and-answer window.

- **The tray is not the engine.** A Windows service cannot show a tray icon or
  a window (Session 0 isolation), but the engine *does* need to run before
  login for reminders to survive a locked machine. So the engine is a boot-time
  scheduled task and the tray is a logon item that attaches to it — starting
  one itself only if nothing answers `/api/health`.
- **`RegisterHotKey`, not a keyboard hook.** The usual library installs a
  low-level global hook — a keylogger by any technical definition, flagged by
  AV and requiring admin. `RegisterHotKey` asks Windows for one chord.
- **The window is pre-warmed.** WebView2 cold start is ~500 ms, which is the
  difference between a reflex and a chore, so it is created hidden at logon and
  the hotkey only calls `show()`.
- `/quick` is a separate page: no hologram, no WebGL, no speech. It runs on its
  own `quick-ask` session so one-off questions do not clutter the HUD
  transcript, while shared ChromaDB means context still carries.

Threads are constrained by the toolkits: `pywebview` must own the main thread,
the Win32 hotkey needs its own `GetMessage` loop, and `pystray` pumps its own.

**Two Windows traps this had to work around, both silent failures:**

1. `pythonw -m uvicorn` **cannot start the engine.** With no console,
   `sys.stderr` is `None`, so uvicorn dies on its first log line with exit 1 —
   before binding, with no console for the traceback to land in. The symptom
   is just "it doesn't start at boot". `run_engine.py` binds real streams to
   `data/logs/engine.log` first, which is why both the boot task and the tray go
   through it.
2. **The engine cannot be stopped with a signal.** It is started with
   `CREATE_NO_WINDOW`, giving it its own console, and `GenerateConsoleCtrlEvent`
   only reaches processes sharing the *caller's* console — which the tray, under
   `pythonw`, does not have. Killing it instead skips lifespan teardown, so
   Telegram's `getUpdates` consumer stays registered and the next start hits
   `409 Conflict`. Shutdown is therefore a token-gated `POST /api/shutdown`,
   with `terminate()` kept only as the fallback for a wedged process.

### H. Frontend
`frontend/` is the **only** directory served at `/`. Serving the project root
previously exposed `gideon.db`, the ChromaDB store, and all source. Audio is
played through a FIFO queue so streamed sentences do not cut each other off.
`quick.html` is served by an explicit `/quick` route registered *before* the
catch-all `StaticFiles` mount, which would otherwise swallow it.

---

## 3. Request Lifecycle

```
[User Input] → POST /api/chat
     │
     ├─ validate session → persist to SQLite → async vector write
     ├─ assemble context (cache-stable ordering, token-budgeted)
     ├─ gate tools on operational intent
     ▼
  Ollama stream
     ├─ native tool call?  → execute → feed result back → re-stream
     ├─ JSON typed as prose? → intercept → treat as tool call
     └─ text → SSE to browser
              └─ on each completed sentence → Kokoro/Piper → SSE audio_url
     ▼
  {"done": true}   (or {"error": ...} — the stream has already begun,
                    so HTTP status can no longer be changed)
```

---

## 4. Repository Map

Dependencies point one way: `interfaces → app → brain / reminders / voice →
tools → storage → core`. Interfaces never import each other; `app.py` is the
only module that knows how the pieces fit.

| Path | Description |
| :--- | :--- |
| `gideon/core/config.py` | Settings only; every value env-overridable |
| `gideon/core/logs.py` | Logging setup and headless stream binding |
| `gideon/core/security.py` | API token, irreversible-command denylist |
| `gideon/core/events.py` | Event bus for proactive notifications |
| `gideon/core/context.py` | Per-turn channel/session (read by tools) |
| `gideon/core/lifecycle.py` | Lets `POST /api/shutdown` stop uvicorn gracefully |
| `gideon/storage/db.py` | Thread-safe SQLite manager + `get_db()` singleton |
| `gideon/storage/memory.py` | `VectorMemory`: ChromaDB, CPU embedder, filtered both ways |
| `gideon/storage/models.py` | Typed rows (`Reminder`, `ChatMessage`) |
| `gideon/storage/remote_scheduler.py` | Fail-soft mirror of reminders to the always-on node |
| `gideon/tools/base.py` | `@tool` / `Param` declarations |
| `gideon/tools/registry.py` | Schemas, ordering, channel policy, dispatch |
| `gideon/brain/` | `prompt.py`, `intent.py`, `toolcalls.py`, `agent.py` |
| `gideon/voice/` | `tts.py` (engines), `streaming.py` (per-sentence speech) |
| `gideon/reminders/service.py` | Delivery, lateness digest, node sync loops |
| `gideon/interfaces/api/` | FastAPI app factory, routes, dependencies |
| `gideon/interfaces/telegram.py` | Telegram bot (sole `getUpdates` consumer) |
| `gideon/interfaces/tray.py` | Tray icon, global hotkey, engine supervision |
| `gideon/interfaces/cli.py` | Terminal chat |
| `gideon/app.py` | Composition root (`Services`) |
| `gideon_tray.py` | **Entry point** — thin launcher for the tray |
| `run_engine.py` | Headless launcher; binds the engine log before uvicorn imports |
| `main.py` | Terminal chat launcher |
| `frontend/` | `index.html`, `app.js`, `index.css`, `quick.html`, hologram `.glb` models |
| `model/` | Kokoro and Piper `.onnx` voices (gitignored) |
| `data/` | Database, vector store, logs, WebView profile (gitignored) |
| `static/audio/` | Generated speech (gitignored, swept on boot) |
| `node/` | Cloudflare Worker + D1 that fires reminders while the PC is off |
| `scripts/migrate_data_dir.py` | Moves pre-restructure state from the root into `data/` |
| `install_autostart.ps1` | Registers the boot task and the logon shortcut |
| `tests/` | pytest suite, isolated from real data by `conftest.py` |
| `run.bat` | Development launcher with 4GB-VRAM Ollama tuning notes |

---

## 5. Running

### Requirements
- Python 3.13+, Ollama with `llama3.2:latest` and `nomic-embed-text`.

### Install once (recommended)
```powershell
# Elevated: registers the engine at boot and the tray at logon
powershell -ExecutionPolicy Bypass -File .\install_autostart.ps1
```
After this there is no command to run. **Ctrl+Alt+G** anywhere summons the
quick-ask window; the tray menu opens the full HUD. Remove it all with
`-Uninstall`.

Two things worth setting alongside it:
- `OLLAMA_KEEP_ALIVE=-1` (or a long value) so the quick window is not waiting
  four seconds for an evicted model to reload. On 4GB this competes with the
  hologram, which is why `/quick` renders no WebGL.
- Ollama's Windows app starts at **logon**, not at boot. Until you log in the
  engine is up and reminders fire, but anything needing the LLM has no model.

### Web server (development)
```powershell
.\run.bat
```
Then open <http://localhost:7777>.

### Terminal CLI
```powershell
.venv\Scripts\python.exe main.py
```

### Tests
```powershell
.venv\Scripts\python.exe -m pytest tests\ -q
```

### Tuning for 4GB VRAM
Set these for the **Ollama service** (not the Python process), then restart
Ollama from the tray:
```powershell
setx OLLAMA_FLASH_ATTENTION 1
setx OLLAMA_KV_CACHE_TYPE q8_0
```
This halves the KV cache (~940MB → ~470MB at 8k context), leaving room for the
3B model and the WebGL hologram to coexist on a 1650 Ti.

### Useful environment variables
| Variable | Default | Effect |
| :--- | :--- | :--- |
| `GIDEON_CHAT_MODEL` | `llama3.2:latest` | Swap the chat model |
| `GIDEON_NUM_CTX` | `8192` | Context window |
| `GIDEON_TOOL_GATING` | `1` | Withhold tools during small talk |
| `GIDEON_MEMORY_MAX_DISTANCE` | `0.75` | Recall relevance threshold |
| `GIDEON_ALLOW_CLOUD_TTS` | `0` | Enable the edge-tts fallback |
| `GIDEON_REQUIRE_CONFIRMATION` | `0` | Gate state-changing tools |
| `GIDEON_PORT` / `GIDEON_HOST` | `7777` / `127.0.0.1` | Bind address |
| `GIDEON_HOTKEY_MODIFIERS` / `_KEY` | `ctrl+alt` / `G` | The global chord |
| `GIDEON_REMINDER_MAX_LATENESS` | `15` | Minutes before a reminder is "missed" |
| `GIDEON_REMINDER_DIGEST_MAX_AGE` | `72` | Hours after which a miss is not worth reporting |
| `GIDEON_REMOTE_SCHEDULER_URL` | — | Always-on node; unset means local-only reminders |
| `GIDEON_REMOTE_SCHEDULER_TOKEN` | — | Shared secret; **both** are required to enable |
| `GIDEON_TTS_ENGINE` | `kokoro` | `kokoro`, falling back to Piper |
| `GIDEON_DATA_DIR` | `data/` | Runtime state; legacy root paths are used until migrated |
