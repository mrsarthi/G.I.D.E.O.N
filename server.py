"""
G.I.D.E.O.N FastAPI Backend

Three things here are load-bearing:

* Speech is synthesised per sentence while the model is still generating, so
  G.I.D.E.O.N starts talking after the first sentence instead of after the
  whole answer.
* /api/events is the always-on channel that lets G.I.D.E.O.N speak first
  (due reminders, hardware warnings).
* Only frontend/ and static/ are served. The old mount exposed the entire
  project directory, gideon.db included.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import time
import uuid
import wave
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Iterator, Optional

import psutil
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import memory
import orchestrator
from config import (
    ALLOWED_HOSTS,
    ALLOW_CLOUD_TTS,
    API_TOKEN,
    API_TOKEN_HEADER,
    AUDIO_DIR,
    AUDIO_RETENTION_HOURS,
    CORS_ORIGINS,
    FRONTEND_DIR,
    MIN_TTS_SENTENCE_CHARS,
    PIPER_MODEL_DIR,
    REMINDER_POLL_SECONDS,
    STATIC_DIR,
    TTS_VOICE_FALLBACK,
    get_logger,
    setup_logging,
)
from db import close_db, get_db
from events import bus

setup_logging()
log = get_logger("gideon.server")

_tts_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="tts_worker")

piper_voice = None
chroma_collection = None

# Sentence boundary: terminator followed by whitespace.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")
_MARKUP = re.compile(r"[*_`#]+")


# ---------------------------------------------------------------------------
#  Text-to-speech
# ---------------------------------------------------------------------------
def init_piper_voice():
    """Load the local Piper ONNX voice into RAM (no VRAM cost)."""
    global piper_voice
    try:
        from piper import PiperVoice
    except ImportError:
        log.warning("piper-tts is not installed; local speech disabled")
        return

    onnx_files = sorted(PIPER_MODEL_DIR.glob("*.onnx"))
    if not onnx_files:
        log.warning("No Piper .onnx voice found in %s", PIPER_MODEL_DIR)
        return

    model_path = onnx_files[0]
    if not model_path.with_suffix(".onnx.json").exists():
        log.warning("Piper voice %s is missing its .json config", model_path.name)
        return

    try:
        log.info("Loading Piper voice %s", model_path.name)
        piper_voice = PiperVoice.load(str(model_path))
        log.info("Piper voice ready (CPU, 0 MB VRAM)")
    except Exception as exc:
        log.error("Piper initialisation failed: %s", exc)


def sweep_audio_dir():
    """Delete generated speech older than the retention window.

    Without this the directory grows without bound; it had reached 31 MB.
    """
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    cutoff = time.time() - (AUDIO_RETENTION_HOURS * 3600)
    removed = 0
    for path in AUDIO_DIR.iterdir():
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


def _clean_for_speech(text: str) -> str:
    return _MARKUP.sub("", text).strip()


def generate_tts_audio(text: str, session_id: str) -> str:
    """Synthesise one utterance and return its served URL ('' on failure)."""
    clean_text = _clean_for_speech(text)
    if not clean_text:
        return ""

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    if piper_voice is not None:
        try:
            filename = f"{session_id}_{uuid.uuid4().hex[:8]}.wav"
            filepath = AUDIO_DIR / filename
            with wave.open(str(filepath), "wb") as wf:
                first = True
                for chunk in piper_voice.synthesize(clean_text):
                    if first:
                        wf.setnchannels(chunk.sample_channels)
                        wf.setsampwidth(chunk.sample_width)
                        wf.setframerate(chunk.sample_rate)
                        first = False
                    wf.writeframes(chunk.audio_int16_bytes)
            return f"/static/audio/{filename}"
        except Exception as exc:
            log.error("Piper synthesis failed: %s", exc)

    if not ALLOW_CLOUD_TTS:
        return ""

    # Opt-in cloud fallback. Off by default: it sends your text to Microsoft.
    try:
        import edge_tts

        filename = f"{session_id}_{uuid.uuid4().hex[:8]}.mp3"
        filepath = AUDIO_DIR / filename

        def _worker():
            loop = asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                comm = edge_tts.Communicate(clean_text, TTS_VOICE_FALLBACK)
                loop.run_until_complete(comm.save(str(filepath)))
            finally:
                loop.close()

        _tts_executor.submit(_worker).result(timeout=15)
        return f"/static/audio/{filename}"
    except Exception as exc:
        log.error("Cloud TTS fallback failed: %s", exc)
        return ""


def split_ready_sentences(buffer: str) -> tuple[list[str], str]:
    """Split a streaming buffer into complete sentences plus a remainder.

    Short fragments are merged forward so a stray "Yes." does not become its
    own audio file.
    """
    parts = _SENTENCE_SPLIT.split(buffer)
    if len(parts) < 2:
        return [], buffer

    *complete, remainder = parts
    sentences: list[str] = []
    carry = ""
    for part in complete:
        candidate = f"{carry} {part}".strip() if carry else part
        if len(candidate) < MIN_TTS_SENTENCE_CHARS:
            carry = candidate
            continue
        sentences.append(candidate)
        carry = ""

    if carry:
        remainder = f"{carry} {remainder}".strip()
    return sentences, remainder


# ---------------------------------------------------------------------------
#  Background daemons
# ---------------------------------------------------------------------------
async def reminder_daemon():
    """Fire due reminders onto the event bus.

    This is what makes set_reminder more than a write-only database call.
    """
    db = get_db()
    log.info("Reminder daemon started (%ds poll)", REMINDER_POLL_SECONDS)
    while True:
        try:
            await asyncio.sleep(REMINDER_POLL_SECONDS)
            due = await asyncio.to_thread(db.get_due_reminders)
            for reminder_id, text, remind_at in due:
                await asyncio.to_thread(db.mark_reminder_completed, reminder_id)
                import random

                honorific = random.choice(["boss", "sir"])
                spoken = f"Reminder {honorific}: {text}"
                audio_url = await asyncio.to_thread(generate_tts_audio, spoken, "reminder")
                bus.publish(
                    "reminder",
                    {
                        "reminder_id": reminder_id,
                        "text": text,
                        "scheduled_for": remind_at,
                        "message": spoken,
                        "audio_url": audio_url,
                    },
                )
                log.info("Fired reminder #%s", reminder_id)
        except asyncio.CancelledError:
            log.info("Reminder daemon stopped")
            raise
        except Exception:
            log.exception("Reminder daemon iteration failed")


from telegram_bot import GideonTelegramDaemon


# ---------------------------------------------------------------------------
#  Lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    global chroma_collection

    bus.bind_loop(asyncio.get_running_loop())
    get_db()
    chroma_collection = memory.initMemory()
    init_piper_voice()
    sweep_audio_dir()

    daemon = asyncio.create_task(reminder_daemon())
    telegram_daemon = GideonTelegramDaemon()
    await telegram_daemon.start()

    log.info("G.I.D.E.O.N online")

    try:
        yield
    finally:
        await telegram_daemon.stop()
        daemon.cancel()
        try:
            await daemon
        except asyncio.CancelledError:
            pass
        close_db()
        _tts_executor.shutdown(wait=False)
        log.info("G.I.D.E.O.N offline")


app = FastAPI(title="G.I.D.E.O.N API Server", lifespan=lifespan)

@app.middleware("http")
async def validate_host_header(request: Request, call_next):
    """Reject requests whose Host is not a loopback name.

    DNS rebinding works by pointing an attacker-controlled domain at
    127.0.0.1; the browser then treats the attacker page as same-origin, so
    CORS does not help. Checking Host does.
    """
    host = (request.headers.get("host") or "").lower()
    if host and host not in ALLOWED_HOSTS:
        log.warning("Rejected request with unexpected Host header: %r", host)
        return JSONResponse(status_code=400, content={"detail": "Invalid Host header."})
    return await call_next(request)


app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


class MessageRequest(BaseModel):
    session_id: str
    message: str


class SessionCreate(BaseModel):
    name: Optional[str] = None


class ProfileUpdateRequest(BaseModel):
    key: str
    value: str


class TerminalExecuteRequest(BaseModel):
    command: str
    cwd: Optional[str] = None


def require_api_token(request: Request) -> None:
    """Gate state-changing endpoints behind the shared secret.

    Without this, any process on the machine could POST to 127.0.0.1 and get
    a shell. The token lives in .env and is handed to the frontend by
    serve_index below.
    """
    supplied = request.headers.get(API_TOKEN_HEADER, "")
    if not secrets.compare_digest(supplied, API_TOKEN):
        log.warning(
            "Rejected unauthenticated %s from %s",
            request.url.path,
            request.client.host if request.client else "unknown",
        )
        raise HTTPException(status_code=401, detail="Missing or invalid API token.")


@app.post("/api/terminal/execute")
async def execute_terminal_cmd(req: TerminalExecuteRequest, request: Request):
    """Execute a manual CLI command from the Terminal HUD drawer."""
    require_api_token(request)

    cmd = req.command.strip()
    if not cmd:
        raise HTTPException(status_code=400, detail="Command string cannot be empty")
    from tools.terminal import run_terminal_command

    result = await asyncio.to_thread(run_terminal_command, cmd, req.cwd, channel="web")
    return {"status": "ok", "command": cmd, "output": result}


@app.get("/api/terminal/audit")
def terminal_audit(request: Request, limit: int = 50):
    """Recent shell-command history, including refused attempts."""
    require_api_token(request)
    return get_db().get_command_audit(limit=min(max(limit, 1), 200))


# ---------------------------------------------------------------------------
#  Sessions
# ---------------------------------------------------------------------------
@app.get("/api/sessions")
def list_sessions():
    try:
        return get_db().list_sessions()
    except Exception as exc:
        log.exception("Listing sessions failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/sessions")
def create_session(data: SessionCreate):
    try:
        name = (data.name or "").strip() or f"Session {datetime.now():%Y-%m-%d %H:%M}"
        session_id = get_db().create_session(name)
        return {"session_id": session_id, "session_name": name}
    except Exception as exc:
        log.exception("Creating session failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: str):
    try:
        get_db().delete_session(session_id)
        if chroma_collection is not None:
            memory.deleteSessionMemories(chroma_collection, session_id)
        return {"status": "success", "message": f"Session {session_id} deleted."}
    except Exception as exc:
        log.exception("Deleting session failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/sessions/{session_id}/history")
def get_session_history(session_id: str):
    try:
        return get_db().get_history(session_id)
    except Exception as exc:
        log.exception("Fetching history failed")
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
#  Profile
# ---------------------------------------------------------------------------
@app.get("/api/profile")
def get_user_profile():
    try:
        return get_db().get_user_profile()
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/profile")
def update_user_profile(data: ProfileUpdateRequest):
    try:
        db = get_db()
        db.set_user_profile(data.key, data.value)
        return {"status": "success", "profile": db.get_user_profile()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
#  Chat
# ---------------------------------------------------------------------------
def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/api/chat")
def chat(data: MessageRequest):
    """Stream a reply as SSE, speaking each sentence as it completes."""
    db = get_db()
    session_id = data.session_id
    message = data.message.strip()

    if not message:
        raise HTTPException(status_code=400, detail="Message content cannot be empty.")
    if not db.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Unknown session_id.")

    user_id = db.insert_message(session_id, "user", message)
    orchestrator.addMemoryAsync(
        chroma_collection, user_id, message, session_id, "user", datetime.now().isoformat()
    )

    def chat_generator() -> Iterator[str]:
        buffer = ""
        pending: list[Future] = []

        def drain(block: bool) -> Iterator[str]:
            """Emit finished audio in submission order."""
            while pending:
                future = pending[0]
                if not block and not future.done():
                    break
                try:
                    url = future.result(timeout=30 if block else None)
                except Exception as exc:
                    log.error("Speech synthesis failed: %s", exc)
                    url = ""
                pending.pop(0)
                if url:
                    yield _sse({"audio_url": url})

        try:
            for token in orchestrator.run_agent_stream(
                db, chroma_collection, session_id, message
            ):
                yield _sse({"text": token})
                buffer += token

                sentences, buffer = split_ready_sentences(buffer)
                for sentence in sentences:
                    # Synthesis runs concurrently with continued generation.
                    pending.append(
                        _tts_executor.submit(generate_tts_audio, sentence, session_id)
                    )

                yield from drain(block=False)

            tail = buffer.strip()
            if tail:
                pending.append(_tts_executor.submit(generate_tts_audio, tail, session_id))

            yield from drain(block=True)
            yield _sse({"done": True})

        except Exception as exc:
            # The response has already started, so an HTTPException here would
            # never reach the client. Send an explicit error event instead.
            log.exception("Chat stream failed")
            yield _sse({"error": str(exc)})
            yield _sse({"done": True})

    return StreamingResponse(
        chat_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
#  Event stream
# ---------------------------------------------------------------------------
@app.get("/api/events")
async def event_stream():
    """Always-on channel for unprompted messages (reminders, alerts)."""
    queue = bus.subscribe()

    async def generator():
        try:
            yield ": connected\n\n"
            while True:
                try:
                    message = await asyncio.wait_for(queue.get(), timeout=20)
                    yield f"data: {message}\n\n"
                except asyncio.TimeoutError:
                    # Comment frame keeps proxies and browsers from timing out.
                    yield ": keepalive\n\n"
        finally:
            bus.unsubscribe(queue)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
#  Metrics
# ---------------------------------------------------------------------------
_last_net_bytes: int | None = None
_last_net_time: float | None = None


def get_network_throughput() -> float:
    global _last_net_bytes, _last_net_time
    try:
        counters = psutil.net_io_counters()
        current_bytes = counters.bytes_sent + counters.bytes_recv
        current_time = time.time()

        if _last_net_bytes is None:
            _last_net_bytes, _last_net_time = current_bytes, current_time
            return 0.1

        elapsed = current_time - _last_net_time
        if elapsed <= 0:
            return 0.1

        mbps = round(((current_bytes - _last_net_bytes) * 8) / (1024 * 1024 * elapsed), 1)
        _last_net_bytes, _last_net_time = current_bytes, current_time
        return max(0.0, mbps)
    except Exception:
        return 0.0


@app.get("/api/metrics")
def get_system_metrics():
    """Real-time host telemetry for the HUD."""
    try:
        cpu_percent = psutil.cpu_percent(interval=None)
        virtual_mem = psutil.virtual_memory()

        gpu_percent = gpu_vram_used = gpu_vram_total = gpu_temp = 0
        try:
            import shutil
            import subprocess

            if shutil.which("nvidia-smi"):
                res = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=5,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                )
                util, used, total, temp = [p.strip() for p in res.strip().split(",")]
                gpu_percent = int(util)
                gpu_vram_used = int(used)
                gpu_vram_total = int(total)
                gpu_temp = int(temp)
        except Exception as exc:
            log.debug("GPU telemetry unavailable: %s", exc)

        start = time.perf_counter()
        get_db().ping()
        latency_ms = max(1, int((time.perf_counter() - start) * 1000))

        drive_status = "Stable"
        try:
            if psutil.disk_usage(".").percent >= 95.0:
                drive_status = "Warning"
        except Exception:
            pass

        return {
            "cpu": cpu_percent,
            "gpu": gpu_percent,
            "gpu_temp": gpu_temp,
            "vram_used": gpu_vram_used,
            "vram_total": gpu_vram_total,
            "ram_used": round(virtual_mem.used / 1024**3, 1),
            "ram_total": round(virtual_mem.total / 1024**3, 1),
            "latency": latency_ms,
            "flux": get_network_throughput(),
            "drive": drive_status,
        }
    except Exception:
        log.exception("Metrics collection failed")
        raise HTTPException(status_code=500, detail="Metrics unavailable")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "piper": piper_voice is not None,
        "vector_documents": chroma_collection.count() if chroma_collection else 0,
        "event_subscribers": bus.subscriber_count,
    }


# ---------------------------------------------------------------------------
#  Static assets
# ---------------------------------------------------------------------------
# Only these two directories are reachable. Serving the project root exposed
# gideon.db, the ChromaDB store and every source file.
STATIC_DIR.mkdir(parents=True, exist_ok=True)
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

if FRONTEND_DIR.is_dir():

    @app.get("/", response_class=HTMLResponse)
    def serve_index():
        """Serve index.html with the API token injected.

        The token reaches the page only through this same-origin route, so a
        cross-origin attacker cannot read it, and the Host check above stops
        DNS rebinding from making itself same-origin.
        """
        html = (FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
        injected = f'<script>window.GIDEON_API_TOKEN = "{API_TOKEN}";</script>'
        if "</head>" in html:
            html = html.replace("</head>", injected + "\n</head>", 1)
        else:
            html = injected + html
        return HTMLResponse(html)

    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
else:  # pragma: no cover - first run before the frontend is moved
    log.error("Frontend directory %s is missing", FRONTEND_DIR)
