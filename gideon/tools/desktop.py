"""
G.I.D.E.O.N Desktop & Telemetry Tools

Application launching is allowlist-only. The previous implementation fell
through to `subprocess.Popen(f"start {name}", shell=True)` with a name the
model chose, which is a shell injection with an LLM holding the pen.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import psutil

from gideon.core.logs import get_logger
from gideon.tools.base import Param, tool

log = get_logger("gideon.tools.desktop")

_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

# The complete set of programs G.I.D.E.O.N may start. Add entries here
# deliberately; nothing outside this table can be launched.
KNOWN_APPS: dict[str, list[str]] = {
    "notepad": ["notepad.exe"],
    "calculator": ["calc.exe"],
    "calc": ["calc.exe"],
    "paint": ["mspaint.exe"],
    "vscode": ["code.cmd", "code"],
    "code": ["code.cmd", "code"],
    "chrome": ["chrome", r"C:\Program Files\Google\Chrome\Application\chrome.exe"],
    "edge": ["msedge", r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"],
    "firefox": ["firefox", r"C:\Program Files\Mozilla Firefox\firefox.exe"],
    "terminal": ["wt.exe", "cmd.exe"],
    "cmd": ["cmd.exe"],
    "powershell": ["powershell.exe"],
    "explorer": ["explorer.exe"],
    "file explorer": ["explorer.exe"],
    "task manager": ["taskmgr.exe"],
    "settings": ["ms-settings:"],
    "spotify": ["spotify.exe"],
    "word": ["winword.exe"],
    "excel": ["excel.exe"],
}

# Roots search_files is permitted to walk.
_SEARCH_ROOTS = [Path.cwd(), Path.home() / "Desktop", Path.home() / "Documents"]
_SKIP_DIRS = {"node_modules", "__pycache__", "venv", ".venv", "AppData", ".git"}
_MAX_MATCHES = 10


@tool(
    "launch_app",
    "ONLY call this when the user explicitly gives a direct command to open or run a desktop application (e.g. 'open notepad', 'launch calc'). NEVER call this to check the time, date, or answer general questions.",
    Param(
        "app_name",
        "Friendly name of the application to start, e.g. 'notepad', 'chrome', 'spotify'.",
        required=True,
        aliases=("app", "name"),
    ),
    # Launching programs is local-only: a remote channel never gets it.
    blocked_channels={"telegram"},
    confirm=True,
    triggers={"open", "launch", "run", "start", "close", "app", "application"},
)
def launch_app(app_name: str) -> str:
    """Start a known desktop application by friendly name."""
    clean_name = (app_name or "").strip().lower()
    targets = KNOWN_APPS.get(clean_name)

    if not targets:
        available = ", ".join(sorted(set(KNOWN_APPS)))
        log.warning("Refused to launch unlisted app: %r", app_name)
        return (
            f"'{app_name}' is not on the approved application list, so I won't launch it. "
            f"I can open: {available}."
        )

    for target in targets:
        try:
            # Protocol handlers (ms-settings:) go through the shell opener,
            # which takes no user-controlled text.
            if target.endswith(":"):
                os.startfile(target)  # noqa: S606 - fixed literal from the allowlist
                return f"Launched {clean_name} successfully."

            if os.path.isabs(target):
                if os.path.exists(target):
                    subprocess.Popen([target], creationflags=_NO_WINDOW)
                    return f"Launched {clean_name} successfully."
                continue

            resolved = shutil.which(target)
            if resolved:
                subprocess.Popen([resolved], creationflags=_NO_WINDOW)
                return f"Launched {clean_name} successfully."
        except Exception as exc:
            log.debug("Launch candidate %r failed: %s", target, exc)
            continue

    return f"'{clean_name}' is approved but I couldn't find it installed on this system."


@tool(
    "get_system_status",
    "Retrieve computer hardware diagnostic telemetry (CPU, GPU, RAM, disk, battery usage). MUST be called whenever the user asks to check system status, hardware, diagnostics, or PC performance.",
    triggers={
        "cpu", "gpu", "ram", "vram", "disk", "drive", "battery", "temperature",
        "temp", "hardware", "diagnostics", "diagnostic", "status", "telemetry",
        "system", "performance", "usage", "specs",
    },
)
def get_system_status() -> str:
    """Report CPU, GPU, RAM, disk and battery telemetry."""
    try:
        cpu = psutil.cpu_percent(interval=0.3)
        mem = psutil.virtual_memory()
        disk = psutil.disk_usage("/")

        lines = [
            f"CPU Utilization: {cpu}%",
            f"GPU Utilization: {_gpu_summary(cpu)}",
            f"RAM Usage: {round(mem.used / 1024**3, 1)} GB / "
            f"{round(mem.total / 1024**3, 1)} GB ({mem.percent}%)",
            f"Primary Disk Free: {round(disk.free / 1024**3, 1)} GB of "
            f"{round(disk.total / 1024**3, 1)} GB",
        ]

        battery = getattr(psutil, "sensors_battery", lambda: None)()
        if battery:
            plugged = "Plugged in" if battery.power_plugged else "On battery"
            lines.append(f"Battery: {battery.percent}% ({plugged})")

        return "\n".join(lines)
    except Exception as exc:
        log.error("System status failed: %s", exc)
        return f"Failed to retrieve system status: {exc}"


def _gpu_summary(cpu_percent: float) -> str:
    """Query nvidia-smi, falling back to an explicit estimate."""
    if not shutil.which("nvidia-smi"):
        return f"~{int(cpu_percent * 0.4)}% (estimated, no NVIDIA driver)"
    try:
        res = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,temperature.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=5,
            creationflags=_NO_WINDOW,
        ).strip()
        util, temp, used, total = [p.strip() for p in res.split(",")]
        return f"{util}% at {temp}°C, VRAM {used}/{total} MB"
    except Exception as exc:
        log.debug("nvidia-smi query failed: %s", exc)
        return f"~{int(cpu_percent * 0.4)}% (estimated)"


@tool(
    "search_files",
    "ONLY call this when the user explicitly commands you to search their local disk for files or folders by filename. NEVER use this for general technical discussion, conceptual research, or summarizing conversations.",
    Param(
        "file_name_or_extension",
        "Filename or extension pattern (e.g. 'main.py', 'report.pdf', '.txt').",
        required=True,
        aliases=("query", "filename", "pattern"),
        kwarg="query",
    ),
    triggers={"file", "files", "folder", "directory", "find", "locate"},
)
def search_files(query: str, search_path: str | None = None) -> str:
    """Find files by name fragment within the approved search roots."""
    clean_query = (query or "").strip().lower()
    if not clean_query:
        return "No filename or extension was provided to search for."

    if search_path:
        roots = [Path(search_path).expanduser().resolve()]
    else:
        roots = [r.resolve() for r in _SEARCH_ROOTS if r.exists()]

    matches: list[str] = []
    for root in roots:
        if len(matches) >= _MAX_MATCHES:
            break
        try:
            for dirpath, dirnames, filenames in os.walk(root):
                dirnames[:] = [
                    d for d in dirnames if not d.startswith(".") and d not in _SKIP_DIRS
                ]
                for name in filenames:
                    if clean_query in name.lower():
                        matches.append(str(Path(dirpath) / name))
                        if len(matches) >= _MAX_MATCHES:
                            break
                if len(matches) >= _MAX_MATCHES:
                    break
        except Exception as exc:
            log.debug("Walk of %s aborted: %s", root, exc)
            continue

    if not matches:
        return f"No files matching '{query}' were found in your usual folders."

    return "\n".join(
        [f"Found {len(matches)} file(s) matching '{query}':"] + [f"— {m}" for m in matches]
    )
