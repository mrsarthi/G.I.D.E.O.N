"""
G.I.D.E.O.N Desktop & Telemetry Tools

Application launching is allowlist-only. The previous implementation fell
through to `subprocess.Popen(f"start {name}", shell=True)` with a name the
model chose, which is a shell injection with an LLM holding the pen.

The allowlist is KNOWN_APPS plus every shortcut in the Windows Start Menu —
programs someone deliberately installed — so new apps work without a code
edit. The model only ever supplies a *name*; what runs is a path this module
found itself, never text the model wrote.
"""

from __future__ import annotations

import difflib
import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

import psutil

from gideon.core.config import SEARCH_ROOTS
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
    "vs code": ["code.cmd", "code"],
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

# ---------------------------------------------------------------------------
#  Start Menu discovery
# ---------------------------------------------------------------------------
_START_MENU_DIRS = [
    Path(os.environ.get("PROGRAMDATA", r"C:\ProgramData")) / r"Microsoft\Windows\Start Menu\Programs",
    Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming")))
    / r"Microsoft\Windows\Start Menu\Programs",
]
# Shortcuts that are not the application itself.
_NOT_AN_APP = re.compile(
    r"uninstall|readme|read me|help|documentation|manual|release notes|website|license|"
    r"changelog|what's new|support",
    re.IGNORECASE,
)
# A friendly name, not a command line: anything else is refused before matching.
_APP_NAME_OK = re.compile(r"^[\w .+\-']{1,60}$")
_SHORTCUT_TTL_SECONDS = 600

_shortcuts: dict[str, Path] = {}
_shortcuts_loaded_at = 0.0
_shortcuts_lock = threading.Lock()


def installed_shortcuts() -> dict[str, Path]:
    """Start Menu shortcuts by lower-cased name, cached for ten minutes."""
    global _shortcuts, _shortcuts_loaded_at
    with _shortcuts_lock:
        if time.monotonic() - _shortcuts_loaded_at < _SHORTCUT_TTL_SECONDS and _shortcuts:
            return _shortcuts
        found: dict[str, Path] = {}
        for root in _START_MENU_DIRS:
            if not root.is_dir():
                continue
            try:
                for lnk in root.rglob("*.lnk"):
                    name = lnk.stem.strip().lower()
                    if name and not _NOT_AN_APP.search(name):
                        found.setdefault(name, lnk)
            except OSError as exc:
                log.debug("Start Menu scan of %s failed: %s", root, exc)
        _shortcuts, _shortcuts_loaded_at = found, time.monotonic()
        log.debug("Indexed %d Start Menu shortcuts", len(found))
        return found


def _match_shortcut(name: str, shortcuts: dict[str, Path]) -> tuple[Path | None, list[str]]:
    """Exact name, else the shortcuts containing every word of `name`.

    Returns (path, []) on a single match, or (None, candidates) when the name
    is ambiguous or unknown, so the caller can ask instead of guessing.
    """
    if name in shortcuts:
        return shortcuts[name], []
    words = name.split()
    candidates = sorted(
        (s for s in shortcuts if all(w in s for w in words)), key=len
    )
    if len(candidates) == 1:
        return shortcuts[candidates[0]], []
    if candidates:
        # "chrome" -> "google chrome" beats "chrome remote desktop": prefer a
        # name that ends with the request, when exactly one does.
        enders = [c for c in candidates if c.endswith(name)]
        if len(enders) == 1:
            return shortcuts[enders[0]], []
        return None, candidates[:5]
    return None, difflib.get_close_matches(name, list(shortcuts), n=3, cutoff=0.75)


def _search_roots() -> list[Path]:
    """Folders search_files walks: configured, or the usual user folders.

    Not the working directory — that is wherever the engine happened to be
    started, which made results depend on how G.I.D.E.O.N was launched.
    """
    if SEARCH_ROOTS:
        return [Path(p).expanduser() for p in SEARCH_ROOTS]
    home = Path.home()
    folders = ("Desktop", "Documents", "Downloads")
    # OneDrive folder backup moves Desktop/Documents under OneDrive.
    candidates = [home / f for f in folders] + [home / "OneDrive" / f for f in folders]
    return [p for p in candidates if p.is_dir()]

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
    privileged=True,
    triggers={"open", "launch", "run", "start", "close", "app", "application"},
)
def launch_app(app_name: str) -> str:
    """Start a built-in or Start Menu application by friendly name."""
    clean_name = " ".join((app_name or "").strip().lower().split())
    targets = KNOWN_APPS.get(clean_name)

    if not targets:
        if not _APP_NAME_OK.match(clean_name):
            log.warning("Refused app name that is not a plain name: %r", app_name)
            return f"'{app_name}' is not on the approved application list, so I won't launch it."

        shortcut, candidates = _match_shortcut(clean_name, installed_shortcuts())
        if shortcut is not None:
            try:
                # The .lnk path came from the Start Menu scan, not the model.
                os.startfile(str(shortcut))  # noqa: S606
                return f"Launched {shortcut.stem} successfully."
            except OSError as exc:
                log.warning("Start Menu shortcut %s failed: %s", shortcut, exc)
                return f"I found {shortcut.stem} in the Start Menu but it failed to start: {exc}"

        log.warning("Refused to launch unlisted app: %r", app_name)
        if candidates:
            return (
                f"'{app_name}' is not on the approved application list as written. "
                f"Did you mean: {', '.join(candidates)}?"
            )
        return (
            f"'{app_name}' is not on the approved application list, so I won't launch it. "
            f"I can open anything in the Start Menu, plus: {', '.join(sorted(set(KNOWN_APPS)))}."
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
        roots = [r.resolve() for r in _search_roots() if r.exists()]

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
