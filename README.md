# G.I.D.E.O.N

**Guided Intelligence Does Everything One Needs** — a local-first personal
assistant for Windows. Chat, voice, tools and reminders run on your own
machine: Ollama for the model, SQLite and ChromaDB for memory, Kokoro or Piper
for speech. Nothing leaves the machine unless you turn on an integration that
needs to (Telegram, the always-on reminder node, cloud TTS).

## What it does

- **Talks back.** Replies stream into a 3D HUD and are spoken sentence by
  sentence while the model is still generating.
- **Does things.** 22 tools: system telemetry, reminders, notes and a to-do
  list, app launching (Start Menu apps), file search, web search, clipboard,
  volume and media keys, YouTube Music, documents, weather, markets, Hacker
  News, IP lookup, Wolfram Alpha and a terminal.
- **Asks before it acts.** Shell commands — and app launches after it has
  read a web page or document — wait for you to click Approve.
- **Is always reachable.** A tray icon and **Ctrl+Alt+G** quick-ask window,
  plus an optional Telegram bot.
- **Remembers.** Conversations persist in SQLite; relevant past messages are
  recalled through vector search.
- **Reminds you even when the PC is off**, via an optional Cloudflare Worker
  (`node/`).

## Quick start

Requirements: Windows, Python 3.13+, [uv](https://docs.astral.sh/uv/), and
[Ollama](https://ollama.com) with `qwen3:4b-instruct` and `nomic-embed-text` pulled
(`ollama pull qwen3:4b-instruct` — the *instruct* tag; plain `qwen3:4b` always
"thinks" and is too slow for conversation).

```powershell
uv sync                                   # install dependencies into .venv
.venv\Scripts\python.exe scratch\download_kokoro_models.py   # optional: Kokoro voice
.\run.bat                                 # development server with reload
```

Open <http://localhost:7777>.

For everyday use, install autostart once (elevated PowerShell). The engine then
starts at boot and the tray at logon:

```powershell
powershell -ExecutionPolicy Bypass -File .\install_autostart.ps1
```

Without admin, add `-TrayOnly` (or tick **Start with Windows** in the tray
menu): the tray then starts at logon and brings the engine up itself. Either
way, G.I.D.E.O.N also appears in the Start Menu, restarts its engine if it
dies, and preloads the model on mains power so the first question is fast.

Other entry points:

| Command | What it starts |
| :--- | :--- |
| `pythonw gideon_tray.py` | Tray icon, global hotkey, engine supervision |
| `pythonw run_engine.py` | Headless engine (what the boot task runs) |
| `python main.py` | Terminal chat |

## Configuration

Settings live in [`gideon/core/config.py`](gideon/core/config.py) and every
one can be overridden by an environment variable or a line in `.env`. The ones
you are most likely to want:

| Variable | Default | Effect |
| :--- | :--- | :--- |
| `GIDEON_CHAT_MODEL` | `qwen3:4b-instruct` | Chat model (chosen by eval — see `FINDINGS.md`) |
| `GIDEON_TTS_ENGINE` | `kokoro` | `kokoro`, falling back to Piper |
| `GIDEON_TELEGRAM_BOT_TOKEN` / `_ALLOWED_USERS` | — | Enable the Telegram bot (both required) |
| `GIDEON_REMOTE_SCHEDULER_URL` / `_TOKEN` | — | Always-on reminder node (both required) |
| `GIDEON_DATA_DIR` | `data/` | Where the database, vector store and logs live |
| `GIDEON_REQUIRE_CONFIRMATION` | `1` | Shell commands from the model need your approval |
| `GIDEON_ON_BATTERY` | `cpu` | On battery the model runs CPU-only (`off` pauses it); the GPU is never used unplugged |
| `GIDEON_TOOL_ROUTER` | `auto` | Offer only relevant tools: on for models that re-read the tool list each turn (llama), off for models that cache it (Qwen) |

See [GIDEON_ARCHITECTURE.md](GIDEON_ARCHITECTURE.md) for the full list.

## Project layout

```
gideon/
  core/         config, logging, security, event bus, turn context, lifecycle
  storage/      SQLite, vector memory, typed models, remote reminder mirror
  tools/        one module per tool family, each declared with @tool
  brain/        prompt assembly, intent heuristics, tool-call parsing, agent loop
  voice/        speech engines and sentence-level streaming
  reminders/    reminder delivery, missed-reminder digest, node sync
  interfaces/   api/ (FastAPI), telegram, tray, cli
  app.py        composition root: builds and wires the services
frontend/       HUD and quick-ask pages
node/           Cloudflare Worker for reminders while the PC is off
scripts/        maintenance (data-directory migration)
tests/          pytest suite
```

Dependencies point one way: `interfaces → app → brain / reminders / voice →
tools → storage → core`.

### What G.I.D.E.O.N may do

`policy.toml` lists every tool with its class (observe / mutate / execute),
whether it needs your approval, and which channels may use it. Edit it and
restart the engine — no code change. A malformed file stops the engine
rather than silently loosening a rule.

### Adding a tool

Declare it next to its implementation and add the module to
`_TOOL_MODULES` in [`gideon/tools/registry.py`](gideon/tools/registry.py):

```python
@tool(
    "get_weather",
    "Fetch live weather ... for a city or local area.",
    Param("location", "City name. Omit for current location.",
          aliases=("city", "place"), default=None),
    triggers={"weather", "forecast", "rain"},
)
def get_weather(location: str | None = None) -> str:
    ...
```

That one declaration produces the schema the model sees, argument
normalisation, the words that make gating and the router offer it
(`triggers`), and — with `grounded=True` on a parameter — the never-assume
check that the value came from the user's own words. Then add the tool to
`policy.toml` with its class and approval rules; the tests fail until you do.

## Measuring it

Every conversation turn is traced (what was asked, which tools ran, how long
each stage took, and your 👍/👎). To measure the model itself:

```powershell
.venv\Scripts\python.exe -m gideon.evals                                   # current model
.venv\Scripts\python.exe -m gideon.evals --models qwen3:4b-instruct,llama3.2:latest
.venv\Scripts\python.exe -m gideon.evals --only clarify,media              # a subset
```

Reports land in `evals/reports/`. Evals use fake tools and a throwaway
database, so they never open apps or touch your data. Run them plugged in —
on battery the model is CPU-only and a full run takes hours.

## Development

```powershell
uv sync --group dev
.venv\Scripts\python.exe -m pytest tests -q
.venv\Scripts\ruff.exe check gideon tests
.venv\Scripts\mypy.exe
.venv\Scripts\pre-commit.exe install      # run the above on every commit
```

The test suite runs against a temporary data directory (see
`tests/conftest.py`), so it never touches your real database or sends to a
configured Telegram bot or reminder node.

### Moving existing data into `data/`

Installs from before the package restructure keep `gideon.db`, `.chromadb`,
`logs` and `.webview` in the project root, and keep working from there. To tidy
up, quit the tray and stop the engine, then:

```powershell
.venv\Scripts\python.exe scripts\migrate_data_dir.py           # dry run
.venv\Scripts\python.exe scripts\migrate_data_dir.py --apply
```
