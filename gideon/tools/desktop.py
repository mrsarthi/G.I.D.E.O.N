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
from dataclasses import dataclass
from pathlib import Path

import psutil

from gideon.core.logs import get_logger
from gideon.storage.db import get_db
from gideon.storage.models import Project
from gideon.storage.projects import (
    default_search_roots,
    get_project_index,
    resolve,
    vscode_last_active_folder,
)
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


_SKIP_DIRS = {"node_modules", "__pycache__", "venv", ".venv", "AppData", ".git"}
_MAX_MATCHES = 10


# ---------------------------------------------------------------------------
#  Opening things in apps: windows and project folders
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AppProfile:
    """What an app can do beyond starting: the flags for each request.

    The model only ever chooses *meanings* (window="new", a project name);
    the command line is built here from these fixed flags and a folder path
    this module looked up itself.
    """

    label: str
    new_window: tuple[str, ...] | None = None  # None: every start is a new window
    reuse_window: tuple[str, ...] | None = None  # None: cannot target an open window
    opens_folders: bool = False


APP_PROFILES: dict[str, AppProfile] = {
    "vscode": AppProfile("VS Code", ("--new-window",), ("--reuse-window",), opens_folders=True),
    "explorer": AppProfile("File Explorer", opens_folders=True),
    "chrome": AppProfile("Chrome", ("--new-window",)),
    "edge": AppProfile("Edge", ("--new-window",)),
    "firefox": AppProfile("Firefox", ("--new-window",)),
}
# Spoken names -> profile.
_PROFILE_NAMES = {
    "vscode": "vscode", "code": "vscode", "vs code": "vscode", "visual studio code": "vscode",
    "explorer": "explorer", "file explorer": "explorer", "files": "explorer",
    "chrome": "chrome", "google chrome": "chrome",
    "edge": "edge", "microsoft edge": "edge",
    "firefox": "firefox",
}
# Characters that never appear in a folder G.I.D.E.O.N would open, and that a
# command interpreter would treat as syntax. Belt and braces: paths come from
# the project index, and Code.exe is started directly, not through code.cmd.
_UNSAFE_PATH = re.compile(r'["%^&|<>\r\n]')


def _vscode_exe() -> str | None:
    """Code.exe itself. `code` is a .cmd script, and arguments passed through a
    batch file are parsed by cmd.exe — a path is safer handed to the exe."""
    candidates = []
    cli = shutil.which("code")
    if cli:
        candidates.append(Path(cli).resolve().parent.parent / "Code.exe")
    for base in (os.environ.get("LOCALAPPDATA"), os.environ.get("PROGRAMFILES")):
        if base:
            candidates += [Path(base) / "Programs" / "Microsoft VS Code" / "Code.exe",
                           Path(base) / "Microsoft VS Code" / "Code.exe"]
    return next((str(c) for c in candidates if c.is_file()), None)


def _app_exe(profile_id: str) -> str | None:
    if profile_id == "vscode":
        return _vscode_exe()
    for target in KNOWN_APPS.get(profile_id, []):
        if os.path.isabs(target) and os.path.exists(target):
            return target
        resolved = shutil.which(target)
        if resolved:
            return resolved
    return None


def _start(argv: list[str]) -> None:
    env = {k: v for k, v in os.environ.items() if k != "ELECTRON_RUN_AS_NODE"}
    subprocess.Popen(argv, env=env, creationflags=_NO_WINDOW)  # noqa: S603 - fixed exe, looked-up path


def _project_lookup(name: str) -> tuple[Project | None, str | None]:
    """(project, None) or (None, what to tell the model instead)."""
    db = get_db()
    projects = db.list_projects()
    if not projects and (index := get_project_index()) is not None:
        index.refresh()  # first use before the background scan has run
        projects = db.list_projects()
    found = resolve(name, projects)
    if found.project is not None:
        return found.project, None
    if found.candidates:
        options = "; ".join(f"{p.name} ({p.path})" for p in found.candidates)
        return None, f"Not opened: '{name}' could mean several folders — {options}. Ask the user which one."
    return None, (
        f"Not opened: I don't know a project or folder called '{name}' "
        f"(I know {len(projects)} under the search folders and in VS Code's history). "
        "Ask the user for the exact folder name."
    )


def _open_project(profile_id: str, project: Project, window: str | None) -> str:
    profile = APP_PROFILES[profile_id]
    if not profile.opens_folders:
        folder_apps = " or ".join(p.label for p in APP_PROFILES.values() if p.opens_folders)
        return f"Not opened: {profile.label} cannot open a folder. {folder_apps} can — ask which the user wants."
    if _UNSAFE_PATH.search(project.path) or not Path(project.path).is_dir():
        return f"Not opened: the folder for {project.name} ({project.path}) is not available."
    exe = _app_exe(profile_id)
    if exe is None:
        return f"Not opened: {profile.label} doesn't seem to be installed, so I couldn't open {project.name}."

    flags: tuple[str, ...] = ()
    where = f"in {profile.label}"
    if window == "new" and profile.new_window is not None:
        flags, where = profile.new_window, f"in a new {profile.label} window"
    elif window == "this":
        if profile.reuse_window is None:
            where = f"in {profile.label} (it opens a new window every time)"
        else:
            flags = profile.reuse_window
            previous = vscode_last_active_folder() if profile_id == "vscode" else None
            replaced = (
                f", replacing {previous.name}"
                if previous and os.path.normcase(str(previous)) != os.path.normcase(project.path)
                else ""
            )
            where = f"in your most recently used {profile.label} window{replaced}"

    try:
        _start([exe, *flags, project.path])
    except OSError as exc:
        return f"Not opened: {profile.label} failed to open {project.name}: {exc}"
    db = get_db()
    db.set_project_opener(project.path, profile.label)
    if (index := get_project_index()) is not None:
        threading.Thread(target=index.refresh_one, args=(project.path,), daemon=True).start()
    log.info("Opened %s %s (%s)", project.path, where, " ".join(flags) or "default window")
    return f"Opened {project.name} ({project.path}) {where}."


def _new_window(profile_id: str) -> str | None:
    """A new, empty window of an app that can make one; None if it can't."""
    profile = APP_PROFILES.get(profile_id)
    if profile is None or profile.new_window is None:
        return None
    exe = _app_exe(profile_id)
    if exe is None:
        return f"Not opened: {profile.label} doesn't seem to be installed."
    try:
        _start([exe, *profile.new_window])
    except OSError as exc:
        return f"Not opened: {profile.label} failed to open a new window: {exc}"
    return f"Opened a new, empty {profile.label} window."


def _app_for_project(project: Project) -> str:
    """The app to open a project in when the user named none."""
    remembered = next(
        (pid for pid, p in APP_PROFILES.items() if p.label == project.opened_with), None
    )
    if remembered is not None:
        return remembered
    return ""


_WINDOW_EVIDENCE = {
    "new": ("new", "another", "fresh", "blank", "empty", "separate", "extra", "second", "instance"),
    "this": ("this", "current", "same", "existing", "reuse"),
}


@tool(
    "launch_app",
    "ONLY call this when the user explicitly gives a direct command to open or run a desktop application, "
    "or to open one of their project folders in an app (e.g. 'open notepad', 'open dumbpipe in vs code'). "
    "NEVER call this to check the time, date, or answer general questions.",
    Param(
        "app_name",
        "Friendly name of the application to start, e.g. 'notepad', 'chrome', 'vs code'.",
        required=True,
        aliases=("app", "name"),
        grounded=True,
        ground_ignore=("app", "application", "program", "open", "launch", "start"),
    ),
    Param(
        "target",
        "A project or folder to open in the app, as the user named it (e.g. 'dumbpipe'). Omit if none.",
        aliases=("project", "folder", "path"),
        default=None,
        grounded=True,
        ground_ignore=("folder", "project", "directory", "repo", "open", "in", "code", "vs"),
    ),
    Param(
        "window",
        "'new' ONLY if the user asks for a new/another window or instance; 'this' ONLY if they say "
        "this/current/same window. Omit otherwise.",
        enum=("new", "this"),
        default=None,
        evidence=_WINDOW_EVIDENCE,
        normalize=lambda v: str(v).strip().lower() if v else None,
    ),
    # Launching programs is local-only: a remote channel never gets it.
    blocked_channels={"telegram"},
    privileged=True,
    triggers={"open", "launch", "run", "start", "close", "app", "application", "window", "instance"},
)
def launch_app(app_name: str, target: str | None = None, window: str | None = None) -> str:
    """Start an app, optionally in a new window or onto one of the user's project folders."""
    # "file_explorer", "vs-code": small models write names as identifiers.
    clean_name = " ".join(re.sub(r"[_-]+", " ", app_name or "").strip().lower().split())
    window = window if window in ("new", "this") else None
    profile_id = _PROFILE_NAMES.get(clean_name)

    is_app = profile_id is not None or clean_name in KNOWN_APPS

    # "open sendme": the project named where the app should be. Small models
    # also send app_name="sendme" with target=<sendme's folder> (seen live).
    if not is_app and _APP_NAME_OK.match(clean_name) and not _match_shortcut(clean_name, installed_shortcuts())[0]:
        named = resolve(clean_name, get_db().list_projects()).project
        if named is not None:
            if target:
                named, problem = _project_lookup(target)
                if named is None:
                    return problem or "Not opened."
            remembered = _app_for_project(named)
            if remembered:
                return _open_project(remembered, named, window) + " (the app you used for it last time)"
            return (
                f"Not opened: no app was named for {named.name}, a project folder ({named.path}). "
                "Ask the user which app to open it in — VS Code or File Explorer."
            )

    if target:
        if not is_app:
            return f"Not opened: {app_name} is not an app I can open folders with. Ask which app to use."
        found, problem = _project_lookup(target)
        if found is None:
            return problem or "Not opened."
        if profile_id is None:
            return f"Not opened: {app_name} cannot open a folder. VS Code or File Explorer can — ask which."
        return _open_project(profile_id, found, window)

    if window == "new" and profile_id is not None:
        made = _new_window(profile_id)
        if made is not None:
            return made
    return _launch(app_name, clean_name)


def _launch(app_name: str, clean_name: str) -> str:
    """Start a built-in or Start Menu application by friendly name."""
    targets = KNOWN_APPS.get(clean_name)

    if not targets:
        if not _APP_NAME_OK.match(clean_name):
            log.warning("Refused app name that is not a plain name: %r", app_name)
            return f"Not opened: '{app_name}' is not on the approved application list, so I won't launch it."

        shortcut, candidates = _match_shortcut(clean_name, installed_shortcuts())
        if shortcut is not None:
            try:
                # The .lnk path came from the Start Menu scan, not the model.
                os.startfile(str(shortcut))  # noqa: S606
                return f"Launched {shortcut.stem} successfully."
            except OSError as exc:
                log.warning("Start Menu shortcut %s failed: %s", shortcut, exc)
                return f"Not opened: I found {shortcut.stem} in the Start Menu but it failed to start: {exc}"

        log.warning("Refused to launch unlisted app: %r", app_name)
        if candidates:
            return (
                f"Not opened: '{app_name}' is not on the approved application list as written. "
                f"Did you mean: {', '.join(candidates)}?"
            )
        return (
            f"Not opened: '{app_name}' is not on the approved application list, so I won't launch it. "
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

    return f"Not opened: '{clean_name}' is approved but I couldn't find it installed on this system."


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
        grounded=True,
        ground_ignore=("file", "folder", "find", "search"),
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
        roots = [r.resolve() for r in default_search_roots() if r.exists()]

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
