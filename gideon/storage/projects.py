"""
G.I.D.E.O.N Project Knowledge (fact memory)

What G.I.D.E.O.N knows about the user's code projects: where each one lives,
what it is, and its state when last looked at — git branch, uncommitted
changes, last commit, when it was last opened in VS Code. This is the first
kind of *fact memory* (PRD §6.2): durable knowledge, as opposed to recall of
past messages.

Where projects come from:

* **VS Code's history** — every folder ever opened has a `workspace.json`
  under `%APPDATA%/Code/User/workspaceStorage`, and the folder's mtime says
  when it was last used. The best signal of what the user actually works on.
* **The search roots** (Desktop, Documents, Downloads unless configured),
  walked a few levels deep for folders that look like projects: `.git`,
  `Cargo.toml`, `package.json`, `pyproject.toml`, …

Where the knowledge goes:

* SQLite `projects` — the source of truth, read by `launch_app` to turn
  "dumbpipe" into a folder, and by the prompt to put a mentioned project's
  facts in front of the model.
* Vector memory — one summary per project, so questions that do not name a
  project ("which of my projects are in Rust?") can still recall them.

Only facts are stored, never file contents, and nothing leaves the machine.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import threading
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import unquote, urlparse

from gideon.core.config import (
    PROJECT_INDEX,
    PROJECT_REFRESH_MINUTES,
    PROJECT_SCAN_DEPTH,
    SEARCH_ROOTS,
)
from gideon.core.logs import get_logger
from gideon.storage.db import DatabaseManager
from gideon.storage.memory import VectorMemory
from gideon.storage.models import Project

log = get_logger("gideon.projects")

_NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

# Marker file -> what kind of project it makes a folder. First match wins,
# so the more specific markers come first.
_MARKERS: tuple[tuple[str, str], ...] = (
    ("Cargo.toml", "Rust"),
    ("go.mod", "Go"),
    ("tsconfig.json", "TypeScript"),
    ("package.json", "JavaScript"),
    ("pyproject.toml", "Python"),
    ("setup.py", "Python"),
    ("requirements.txt", "Python"),
    ("pom.xml", "Java"),
    ("build.gradle", "Java/Kotlin"),
    ("CMakeLists.txt", "C/C++"),
    ("composer.json", "PHP"),
    ("Gemfile", "Ruby"),
    ("deno.json", "Deno"),
    ("pubspec.yaml", "Dart/Flutter"),
)
_MARKER_NAMES = {m for m, _ in _MARKERS} | {".git"}
_SKIP_DIRS = {
    "node_modules", "__pycache__", "venv", ".venv", "env", "AppData", "target",
    "dist", "build", "site-packages", "$RECYCLE.BIN",
}
_README_NAMES = ("README.md", "README.rst", "README.txt", "README", "readme.md")
_GIT_TIMEOUT = 5
_DESCRIPTION_CHARS = 160


def default_search_roots() -> list[Path]:
    """Configured roots, or Desktop/Documents/Downloads (OneDrive-backed too).

    Not the working directory — that is wherever the engine happened to be
    started, which made results depend on how G.I.D.E.O.N was launched.
    """
    if SEARCH_ROOTS:
        return [Path(p).expanduser() for p in SEARCH_ROOTS]
    home = Path.home()
    folders = ("Desktop", "Documents", "Downloads")
    candidates = [home / f for f in folders] + [home / "OneDrive" / f for f in folders]
    return [p for p in candidates if p.is_dir()]


def name_key(text: str) -> str:
    """How project names are compared: "Dumb Pipe", "dumb-pipe" -> "dumbpipe"."""
    return re.sub(r"[^a-z0-9]", "", text.lower())


# ---------------------------------------------------------------------------
#  Discovery
# ---------------------------------------------------------------------------
def _vscode_user_dir() -> Path:
    return Path(os.environ.get("APPDATA", str(Path.home() / "AppData" / "Roaming"))) / "Code" / "User"


def _folder_from_uri(uri: str) -> Path | None:
    """file:///c%3A/Users/x/proj -> C:\\Users\\x\\proj. Remote URIs are skipped."""
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        return None
    path = unquote(parsed.path)
    if re.match(r"^/[A-Za-z]:", path):  # "/c:/Users" on Windows
        path = path[1].upper() + path[2:]  # and "C:", as Explorer spells it
    return Path(path)


def vscode_history(user_dir: Path | None = None) -> dict[Path, datetime]:
    """Folders opened in VS Code, with when each was last used."""
    storage = (user_dir or _vscode_user_dir()) / "workspaceStorage"
    # Keyed case-insensitively (Windows paths), keeping the real spelling.
    found: dict[str, tuple[Path, datetime]] = {}
    if not storage.is_dir():
        return {}
    for entry in storage.iterdir():
        try:
            data = json.loads((entry / "workspace.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        folder = _folder_from_uri(str(data.get("folder") or ""))
        if folder is None or folder.name.startswith(".") or not folder.is_dir():
            continue
        used = datetime.fromtimestamp(entry.stat().st_mtime)
        key = os.path.normcase(str(folder))
        if key not in found or used > found[key][1]:
            found[key] = (folder, used)
    return dict(found.values())


def vscode_last_active_folder(user_dir: Path | None = None) -> Path | None:
    """The folder in the VS Code window used last (what "this window" holds)."""
    try:
        data = json.loads(
            ((user_dir or _vscode_user_dir()) / "globalStorage" / "storage.json").read_text(encoding="utf-8")
        )
        uri = data["windowsState"]["lastActiveWindow"]["folder"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return _folder_from_uri(str(uri))


def _looks_like_project(path: Path) -> bool:
    try:
        return any((path / marker).exists() for marker in _MARKER_NAMES)
    except OSError:
        return False


def scan_roots(roots: Iterable[Path], depth: int = PROJECT_SCAN_DEPTH) -> Iterator[Path]:
    """Project folders under `roots`. A project's own subfolders are not searched."""
    for root in roots:
        yield from _walk(root, depth)


def _walk(folder: Path, depth: int) -> Iterator[Path]:
    if _looks_like_project(folder):
        yield folder
        return
    if depth <= 0:
        return
    try:
        children = sorted(
            (e for e in os.scandir(folder) if e.is_dir(follow_symlinks=False)), key=lambda e: e.name
        )
    except OSError:
        return
    for child in children:
        if child.name.startswith(".") or child.name in _SKIP_DIRS:
            continue
        yield from _walk(Path(child.path), depth - 1)


# ---------------------------------------------------------------------------
#  Inspection
# ---------------------------------------------------------------------------
def _git(path: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(path), *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=_GIT_TIMEOUT,
            creationflags=_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 else None


def _kind(path: Path) -> str | None:
    for marker, kind in _MARKERS:
        if (path / marker).exists():
            return kind
    return None


def _description(path: Path) -> str | None:
    """The README's first paragraph of prose: no headings, badges or HTML.

    A paragraph, not a line: READMEs wrap mid-sentence ("This is an example
    application using iroh with" was sendme's first line).
    """
    for name in _README_NAMES:
        readme = path / name
        if not readme.is_file():
            continue
        try:
            with readme.open(encoding="utf-8", errors="replace") as handle:
                lines = [line.strip() for _, line in zip(range(80), handle, strict=False)]
        except OSError:
            continue
        paragraph: list[str] = []
        for text in lines:
            if text.startswith(("#", "[!", "![", "<", "=", "-", "|", "```")):
                text = ""
            if not text:
                if paragraph:
                    break
                continue
            paragraph.append(text)
        if not paragraph:
            continue
        prose = " ".join(paragraph)
        prose = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", prose)  # links -> their text
        prose = re.sub(r"[*_`]", "", prose)
        if len(prose) <= _DESCRIPTION_CHARS:
            return prose
        cut = prose[:_DESCRIPTION_CHARS].rsplit(" ", 1)[0].rstrip(",;:")
        return cut + "…"
    return None


def _modified_at(path: Path) -> datetime | None:
    """Newest top-level change (git's index counts: it moves on every stage)."""
    stamps = []
    try:
        for entry in os.scandir(path):
            if entry.name in _SKIP_DIRS:
                continue
            stamps.append(entry.stat(follow_symlinks=False).st_mtime)
    except OSError:
        return None
    index = path / ".git" / "index"
    if index.exists():
        stamps.append(index.stat().st_mtime)
    return datetime.fromtimestamp(max(stamps)) if stamps else None


def _stamp(value: datetime | None) -> str | None:
    return value.isoformat(timespec="minutes") if value else None


def inspect(path: Path, last_opened: datetime | None = None, now: datetime | None = None) -> Project:
    """Everything worth knowing about one project, right now."""
    now = now or datetime.now()
    project = Project(path=str(path), name=path.name, kind=_kind(path), description=_description(path))
    project.modified_at = _stamp(_modified_at(path))
    project.last_opened_at = _stamp(last_opened)

    if (path / ".git").exists():
        status = _git(path, "status", "--porcelain=v1", "--branch")
        if status is not None:
            lines = status.splitlines()
            head = lines[0][3:] if lines and lines[0].startswith("## ") else ""
            project.branch = head.split("...")[0].replace("No commits yet on ", "") or None
            project.dirty = sum(1 for line in lines[1:] if line.strip())
        log_line = _git(path, "log", "-1", "--format=%cI%x1f%s")
        if log_line and "\x1f" in log_line:
            when, subject = log_line.strip().split("\x1f", 1)
            project.last_commit_at = when[:16]
            project.last_commit = subject[:120]

    project.refreshed_at = now.isoformat(timespec="minutes")
    project.summary = summarize(project)
    return project


def _day(stamp: str | None) -> str | None:
    return stamp[:10] if stamp else None


def summarize(project: Project) -> str:
    """One self-contained paragraph: what the model reads, and what gets embedded."""
    # A folder opened in VS Code without project markers ("Scripts") is still
    # worth knowing, but it is not a project.
    label = "Project" if project.kind or project.branch is not None else "Folder"
    parts = [f"{label} {project.name} — folder {project.path}."]
    if project.kind:
        parts.append(f"{project.kind} project.")
    if project.description:
        parts.append(f"About: {project.description}")
    if project.branch is not None:
        changes = (
            "no uncommitted changes" if not project.dirty
            else f"{project.dirty} uncommitted change{'s' if project.dirty != 1 else ''}"
        )
        parts.append(f"Git branch {project.branch}, {changes}.")
        if project.last_commit:
            parts.append(f'Last commit {_day(project.last_commit_at)}: "{project.last_commit}".')
    elif project.kind:
        parts.append("Not a git repository.")
    if project.modified_at:
        parts.append(f"Files last changed {_day(project.modified_at)}.")
    if project.last_opened_at:
        parts.append(f"Last opened in VS Code {_day(project.last_opened_at)}.")
    if project.opened_with:
        parts.append(f"The user opens it with {project.opened_with}.")
    parts.append(f"(State as of {project.refreshed_at.replace('T', ' ')}.)")
    return " ".join(parts)


# ---------------------------------------------------------------------------
#  Lookup
# ---------------------------------------------------------------------------
@dataclass
class Resolution:
    """What a spoken project name turned out to mean."""

    project: Project | None = None
    candidates: list[Project] = field(default_factory=list)


def resolve(name: str, projects: list[Project]) -> Resolution:
    """Find the project the user named, or the candidates when unsure.

    Exact name first ("dumbpipe", "dumb pipe", "Dumb-Pipe"), then names that
    start with or contain what was said. A path the model copied from the
    project list resolves to that project. Never a fuzzy guess: a near miss
    comes back as candidates, so the caller asks.
    """
    said = (name or "").strip().strip("\"'")
    if not said:
        return Resolution()
    by_path = {os.path.normcase(p.path): p for p in projects}
    if os.path.normcase(said) in by_path:
        return Resolution(project=by_path[os.path.normcase(said)])

    key = name_key(Path(said).name if ("\\" in said or "/" in said) else said)
    if not key:
        return Resolution()
    exact = [p for p in projects if name_key(p.name) == key]
    if len(exact) == 1:
        return Resolution(project=exact[0])
    if exact:
        return Resolution(candidates=exact)
    partial = [p for p in projects if key in name_key(p.name)]
    if len(partial) == 1 and len(key) >= 4:
        return Resolution(project=partial[0])
    return Resolution(candidates=partial[:5])


def mentioned_in(text: str, projects: list[Project]) -> list[Project]:
    """Projects whose name appears in `text` as a word ("open dumbpipe in…")."""
    words = re.findall(r"[a-z0-9][a-z0-9._-]*", text.lower())
    keys = {name_key(w) for w in words}
    # Two-word names said with a space: "dumb pipe".
    keys |= {name_key(a + b) for a, b in zip(words, words[1:], strict=False)}
    return [p for p in projects if len(name_key(p.name)) >= 3 and name_key(p.name) in keys]


# ---------------------------------------------------------------------------
#  The index
# ---------------------------------------------------------------------------
class ProjectIndex:
    """Keeps the `projects` table and their fact-memory summaries current."""

    def __init__(
        self,
        db: DatabaseManager,
        memory: VectorMemory | None,
        roots: list[Path] | None = None,
        vscode_dir: Path | None = None,
    ) -> None:
        self.db = db
        self.memory = memory
        self._roots = roots
        self._vscode_dir = vscode_dir
        self._lock = threading.Lock()

    def refresh(self) -> int:
        """Rescan everything. Returns how many projects are known."""
        with self._lock:
            started = datetime.now()
            history = vscode_history(self._vscode_dir)
            folders: dict[str, tuple[Path, datetime | None]] = {}
            for folder in scan_roots(self._roots if self._roots is not None else default_search_roots()):
                folders[os.path.normcase(str(folder))] = (folder, None)
            for folder, used in history.items():
                key = os.path.normcase(str(folder))
                # VS Code history is kept even for folders that do not look
                # like projects: the user opened them as one.
                folders[key] = (folders.get(key, (folder, None))[0], used)

            previous = {os.path.normcase(p.path): p for p in self.db.list_projects(include_missing=True)}
            for key, (folder, opened) in folders.items():
                project = inspect(folder, opened, now=started)
                old = previous.get(key)
                if old is not None:
                    project.path = old.path  # keep the stored spelling
                    project.opened_with = old.opened_with
                    project.summary = summarize(project)
                self.db.upsert_project(project)
                self._remember(project, old)

            stored = {os.path.normcase(p.path): p.path for p in self.db.list_projects(include_missing=True)}
            gone = self.db.mark_projects_missing({stored[k] for k in folders if k in stored})
            for path in gone:
                if self.memory is not None:
                    self.memory.delete_fact(_fact_id(path))
            log.info(
                "Project index: %d projects (%d from VS Code), %d gone, in %.1fs",
                len(folders), len(history), len(gone), (datetime.now() - started).total_seconds(),
            )
            return len(folders)

    def refresh_one(self, path: str) -> Project | None:
        """Re-inspect one project now (after opening it, say)."""
        with self._lock:
            current = {os.path.normcase(p.path): p for p in self.db.list_projects(include_missing=True)}
            old = current.get(os.path.normcase(path))
            if old is None or not Path(path).is_dir():
                return None
            used = datetime.fromisoformat(old.last_opened_at) if old.last_opened_at else None
            project = inspect(Path(old.path), used)
            project.opened_with = old.opened_with
            project.summary = summarize(project)
            self.db.upsert_project(project)
            self._remember(project, old)
            return project

    def _remember(self, project: Project, old: Project | None) -> None:
        """Write the summary to fact memory when it says something new.

        The "as of" stamp is left out of the comparison, so an unchanged
        project is not re-embedded every refresh; it is re-embedded at least
        once a day so the stamp the model reads stays roughly right.
        """
        if self.memory is None:
            return
        if (
            old is not None
            and not old.missing
            and _without_stamp(old.summary) == _without_stamp(project.summary)
            and _day(old.refreshed_at) == _day(project.refreshed_at)
        ):
            return
        self.memory.upsert_fact(_fact_id(project.path), project.summary, "project", project.refreshed_at)

    async def run_refresh_loop(self, first_delay: float = 60.0) -> None:
        """Refresh after start-up settles, then every PROJECT_REFRESH_MINUTES."""
        if not PROJECT_INDEX:
            return
        await asyncio.sleep(first_delay)  # leave the start for the model warm-up
        while True:
            try:
                await asyncio.to_thread(self.refresh)
            except Exception:
                log.exception("Project index refresh failed")
            await asyncio.sleep(max(PROJECT_REFRESH_MINUTES, 1) * 60)


def _fact_id(path: str) -> str:
    return "project:" + os.path.normcase(path).replace("\\", "/")


def _without_stamp(summary: str) -> str:
    return re.sub(r"\(State as of [^)]*\)", "", summary or "")


# Set by the composition root; tools read it through get_project_index().
_index: ProjectIndex | None = None


def set_project_index(index: ProjectIndex | None) -> None:
    global _index
    _index = index


def get_project_index() -> ProjectIndex | None:
    return _index
