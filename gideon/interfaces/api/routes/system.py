"""Engine control and host telemetry: terminal, shutdown, metrics, health."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import time

import psutil
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from gideon.app import Services
from gideon.core import lifecycle
from gideon.core.config import BATTERY_MODE
from gideon.core.logs import get_logger
from gideon.core.power import battery_percent, on_battery
from gideon.interfaces.api.deps import get_services, require_api_token
from gideon.tools.terminal import run_terminal_command

log = get_logger("gideon.api.system")

router = APIRouter()


class TerminalExecuteRequest(BaseModel):
    command: str
    cwd: str | None = None


# ---------------------------------------------------------------------------
#  Token-gated control
# ---------------------------------------------------------------------------
@router.post("/api/terminal/execute", dependencies=[Depends(require_api_token)])
async def execute_terminal_cmd(req: TerminalExecuteRequest):
    """Execute a manual CLI command from the Terminal HUD drawer."""
    cmd = req.command.strip()
    if not cmd:
        raise HTTPException(status_code=400, detail="Command string cannot be empty")
    result = await asyncio.to_thread(run_terminal_command, cmd, req.cwd, channel="web")
    return {"status": "ok", "command": cmd, "output": result}


@router.get("/api/terminal/audit", dependencies=[Depends(require_api_token)])
def terminal_audit(limit: int = 50, services: Services = Depends(get_services)):
    """Recent shell-command history, including refused attempts."""
    return services.db.get_command_audit(limit=min(max(limit, 1), 200))


@router.post("/api/shutdown", dependencies=[Depends(require_api_token)])
def request_shutdown():
    """Stop the engine gracefully so lifespan teardown actually runs.

    Token-gated and loopback-only (the Host middleware). Strictly less
    dangerous than /api/terminal/execute, which is gated the same way.

    This exists because the tray cannot signal the engine: it starts it with
    CREATE_NO_WINDOW, giving the child its own console, and CTRL_BREAK only
    travels within a shared console. Killing it instead skips lifespan, which
    leaves Telegram's getUpdates consumer registered — the next start then
    fails with 409 Conflict.
    """
    if not lifecycle.request_shutdown():
        # Started via run.bat or `uvicorn` directly; the caller has to fall
        # back to terminating the process itself.
        raise HTTPException(
            status_code=503,
            detail="No uvicorn server bound; engine was not started by run_engine.py.",
        )
    return {"status": "stopping"}


# ---------------------------------------------------------------------------
#  Telemetry
# ---------------------------------------------------------------------------
class _NetworkMeter:
    """Throughput between successive /api/metrics polls."""

    def __init__(self) -> None:
        self._last_bytes: int | None = None
        self._last_time: float | None = None

    def mbps(self) -> float:
        try:
            counters = psutil.net_io_counters()
            current_bytes = counters.bytes_sent + counters.bytes_recv
            current_time = time.time()

            if self._last_bytes is None or self._last_time is None:
                self._last_bytes, self._last_time = current_bytes, current_time
                return 0.1

            elapsed = current_time - self._last_time
            if elapsed <= 0:
                return 0.1

            mbps = round(((current_bytes - self._last_bytes) * 8) / (1024 * 1024 * elapsed), 1)
            self._last_bytes, self._last_time = current_bytes, current_time
            return max(0.0, mbps)
        except Exception:
            return 0.0


_network = _NetworkMeter()


def _gpu_stats() -> tuple[int, int, int, int]:
    """(utilisation %, VRAM used MB, VRAM total MB, temperature C); zeros if unknown."""
    try:
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
            return int(util), int(used), int(total), int(temp)
    except Exception as exc:
        log.debug("GPU telemetry unavailable: %s", exc)
    return 0, 0, 0, 0


@router.get("/api/metrics")
def get_system_metrics(services: Services = Depends(get_services)):
    """Real-time host telemetry for the HUD."""
    try:
        cpu_percent = psutil.cpu_percent(interval=None)
        virtual_mem = psutil.virtual_memory()
        gpu_percent, gpu_vram_used, gpu_vram_total, gpu_temp = _gpu_stats()

        start = time.perf_counter()
        services.db.ping()
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
            "flux": _network.mbps(),
            "drive": drive_status,
            # The HUD throttles its WebGL hologram on battery.
            "on_battery": on_battery(),
            "battery": battery_percent(),
        }
    except Exception as exc:
        log.exception("Metrics collection failed")
        raise HTTPException(status_code=500, detail="Metrics unavailable") from exc


@router.get("/api/health")
def health(services: Services = Depends(get_services)):
    engines = services.speech.engines
    return {
        "status": "ok",
        "piper": engines["piper"],
        "kokoro": engines["kokoro"],
        "vector_documents": services.memory.count(),
        "event_subscribers": services.events.subscriber_count,
        "on_battery": on_battery(),
        "battery_mode": BATTERY_MODE,
    }
