# G.I.D.E.O.N — System Architecture & Technical Specification

> **G.I.D.E.O.N** (*Guided Intelligence Does Everything One Needs*) is a privacy-first, local-first AI assistant built on **Ollama**, **FastAPI**, **SQLite**, **ChromaDB**, **Piper TTS**, and **Three.js WebGL**.

---

## 1. High-Level Architecture

```
                  ┌─────────────────────────────────────────┐
                  │      3D Holographic Web Interface       │
                  │   Three.js WebGL + Glassmorphism HUD    │
                  └───────┬────────────────────────┬────────┘
                          │ SSE /api/chat          │ SSE /api/events
                          ▼                        ▼
                  ┌─────────────────────────────────────────┐
                  │          FastAPI Web Backend            │
                  │              (server.py)                │
                  │   sentence-streaming TTS · telemetry    │
                  └───────┬────────────────────────┬────────┘
                          │                        │
                          ▼                        ▼
          ┌───────────────────────────┐  ┌──────────────────────┐
          │   Orchestration Engine    │  │   Event Bus          │
          │     (orchestrator.py)     │  │   (events.py)        │
          │  cache-stable prompt      │  │   reminder daemon    │
          │  tool gating · tool loop  │  │   proactive speech   │
          └────┬─────────────────┬────┘  └──────────────────────┘
               │                 │
               ▼                 ▼
   ┌───────────────────┐  ┌──────────────────────┐
   │  Tool Registry    │  │  Local LLM (Ollama)  │
   │  (tools/*.py)     │  │  llama3.2 · GPU      │
   └───────────────────┘  └──────────────────────┘
               │
        ┌──────┴────────────────┐
        ▼                       ▼
┌──────────────────┐  ┌──────────────────────────┐
│ SQLite (db.py)   │  │ ChromaDB (memory.py)     │
│ gideon.db / WAL  │  │ nomic-embed-text · CPU   │
└──────────────────┘  └──────────────────────────┘
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
- **Token budget** (`config.py`): `num_ctx` minus the tool schemas Ollama
  prepends (~1,220 tokens), minus the memory reserve, minus room for the
  answer. Overflowing this silently drops the system prompt off the front.
- **Tool gating**: tool schemas are only offered for messages with operational
  intent. Small models otherwise fire tools during small talk — observed:
  *"say thanks in one short sentence"* invoked `set_clipboard`.
- **Pseudo tool-call interception**: replies that open with a JSON object are
  withheld until they can be classified. Models frequently type
  `{"name": ..., "parameters": ...}` as prose; unintercepted it reaches the UI
  and gets read aloud.

### B. Dual-Tier Memory
1. **Relational (SQLite — `db.py`)** — sessions, messages, profile, reminders.
   One shared connection behind an `RLock`, WAL mode, `busy_timeout`, and a
   checkpoint on close. Use `get_db()`; constructing `DatabaseManager` per call
   site opens redundant connections to the same file.
2. **Semantic (ChromaDB — `memory.py`)** — embeddings via `nomic-embed-text`,
   **pinned to CPU**. On a 4GB card a GPU embedder evicts the resident chat
   model on every write. Writes are filtered (filler and tool output are never
   indexed) and reads are distance-thresholded, so unrelated memories stop
   being injected into every turn.

### C. Tools (`tools/registry.py`)
Native Ollama JSON function calling. Schemas and handlers are kept in sync by
deriving `VALID_TOOL_NAMES` from the schema list.

| Tool | Purpose |
| :--- | :--- |
| `get_system_status` | CPU/GPU/RAM/disk/battery telemetry |
| `set_reminder` / `list_reminders` | Scheduling, fired by the reminder daemon |
| `launch_app` | **Allowlist-only** application launcher |
| `search_files` | Filename search across approved roots |
| `search_web` | Headless DDG search with Instant-Answer/Wikipedia fallbacks |
| `get_clipboard` / `set_clipboard` | Windows clipboard access |
| `control_volume` | Master volume via pycaw |
| `control_media` | Play/pause/next/previous via media keys |
| `read_document` | PDF and text/code extraction |

`launch_app` accepts only names present in `KNOWN_APPS`. The previous
fall-through to `subprocess.Popen(f"start {name}", shell=True)` was a shell
injection with the model holding the pen.

### D. Voice (Piper, local)
- **Primary**: Piper ONNX on CPU — 0 MB VRAM.
- **Fallback**: `edge-tts`, **off by default** (`GIDEON_ALLOW_CLOUD_TTS=1` to
  enable) because it sends text to Microsoft.
- **Sentence streaming**: speech is synthesised per completed sentence while
  the model is still generating, and emitted in order. Measured: first audio at
  8.96s on a reply that finished at 12.4s.
- Generated audio is swept at startup (`GIDEON_AUDIO_RETENTION_HOURS`).

### E. Event Bus (`events.py`)
A fan-out pub/sub over asyncio queues backing `GET /api/events`. The reminder
daemon polls for due rows, marks them completed, synthesises speech, and
publishes — this is what makes `set_reminder` more than a database write.
Saturated subscribers drop events rather than stalling publishers.

### F. Frontend
`frontend/` is the **only** directory served at `/`. Serving the project root
previously exposed `gideon.db`, the ChromaDB store, and all source. Audio is
played through a FIFO queue so streamed sentences do not cut each other off.

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
              └─ on each completed sentence → Piper → SSE audio_url
     ▼
  {"done": true}   (or {"error": ...} — the stream has already begun,
                    so HTTP status can no longer be changed)
```

---

## 4. Repository Map

| Path | Description |
| :--- | :--- |
| `config.py` | Central configuration and logging; every value env-overridable |
| `server.py` | FastAPI app, SSE endpoints, streaming TTS, reminder daemon |
| `orchestrator.py` | Prompt assembly, tool gating, agent loop |
| `events.py` | Event bus for proactive notifications |
| `db.py` | Thread-safe SQLite manager + `get_db()` singleton |
| `memory.py` | ChromaDB vector memory (CPU embedder, filtered both ways) |
| `tools/` | Tool implementations and the registry |
| `frontend/` | `index.html`, `app.js`, `index.css`, hologram `.glb` models |
| `model/` | Piper `.onnx` voice (gitignored — 60MB+) |
| `static/audio/` | Generated speech (gitignored, swept on boot) |
| `tests/` | pytest suite (98 tests) |
| `main.py` | Terminal CLI entry point |
| `run.bat` | Launcher with 4GB-VRAM Ollama tuning notes |

---

## 5. Running

### Requirements
- Python 3.13+, Ollama with `llama3.2:latest` and `nomic-embed-text`.

### Web server
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
