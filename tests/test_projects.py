"""Project knowledge (fact memory) and opening projects/windows in apps."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gideon.brain.prompt import build_live_context_block, project_facts
from gideon.storage import projects
from gideon.storage.db import get_db
from gideon.storage.models import Project
from gideon.storage.projects import ProjectIndex, mentioned_in, resolve, scan_roots, vscode_history
from gideon.tools import desktop
from gideon.tools.registry import get_tool
from gideon.tools.validation import check_call


@pytest.fixture(autouse=True)
def _no_projects():
    db = get_db()
    with db.conn:
        db.conn.execute("DELETE FROM projects")
    yield
    with db.conn:
        db.conn.execute("DELETE FROM projects")


def _project(tmp_path: Path, name: str, marker: str = "Cargo.toml") -> Path:
    folder = tmp_path / name
    folder.mkdir(parents=True)
    (folder / marker).write_text("")
    return folder


class FakeMemory:
    def __init__(self):
        self.facts: dict[str, str] = {}
        self.writes = 0

    def upsert_fact(self, fact_id, text, kind, updated_at):
        self.facts[fact_id] = text
        self.writes += 1
        return True

    def delete_fact(self, fact_id):
        self.facts.pop(fact_id, None)


# ---------------------------------------------------------------------------
#  Discovery
# ---------------------------------------------------------------------------
def test_scan_finds_projects_but_not_inside_them(tmp_path):
    _project(tmp_path, "OPEN SOURCE/dumbpipe")
    _project(tmp_path, "OPEN SOURCE/dumbpipe/examples/inner", "package.json")  # part of dumbpipe
    _project(tmp_path, "web/node_modules/lib", "package.json")  # dependency, not a project
    (tmp_path / "notes").mkdir()  # no marker

    found = sorted(p.name for p in scan_roots([tmp_path]))
    assert found == ["dumbpipe"]


def test_vscode_history_reads_workspace_storage(tmp_path):
    folder = _project(tmp_path, "OPEN SOURCE/sendme")
    entry = tmp_path / "vscode" / "workspaceStorage" / "abc123"
    entry.mkdir(parents=True)
    uri = "file:///" + str(folder).replace("\\", "/").replace(":", "%3A").replace(" ", "%20")
    (entry / "workspace.json").write_text(json.dumps({"folder": uri.replace("C%3A", "c%3A")}))
    remote = tmp_path / "vscode" / "workspaceStorage" / "remote"
    remote.mkdir()
    (remote / "workspace.json").write_text(json.dumps({"folder": "vscode-remote://ssh/home/x"}))

    history = vscode_history(tmp_path / "vscode")
    assert [p.name for p in history] == ["sendme"]
    assert str(next(iter(history)))[0].isupper()  # "C:\", not VS Code's "c:"


# ---------------------------------------------------------------------------
#  Lookup
# ---------------------------------------------------------------------------
def _known(*names_and_paths: tuple[str, str]) -> list[Project]:
    return [Project(path=path, name=name) for name, path in names_and_paths]


KNOWN = _known(
    ("dumbpipe", r"C:\code\dumbpipe"),
    ("sendme", r"C:\code\sendme"),
    ("G.I.D.E.O.N", r"C:\code\G.I.D.E.O.N"),
    ("iroh-blobs", r"C:\code\iroh-blobs"),
    ("resilient-pasteur", r"C:\a\resilient-pasteur"),
    ("resilient-pasteur", r"C:\b\resilient-pasteur"),
)


@pytest.mark.parametrize(
    "said,expected",
    [
        ("dumbpipe", "dumbpipe"),
        ("dumb pipe", "dumbpipe"),
        ("Dumb-Pipe", "dumbpipe"),
        ("gideon", "G.I.D.E.O.N"),
        ("iroh", "iroh-blobs"),  # the only project containing it
        (r"C:\code\sendme", "sendme"),  # a path copied from the facts
    ],
)
def test_resolve(said, expected):
    assert resolve(said, KNOWN).project.name == expected


def test_two_projects_with_one_name_are_candidates_not_a_guess():
    found = resolve("resilient pasteur", KNOWN)
    assert found.project is None
    assert len(found.candidates) == 2


def test_unknown_name_resolves_to_nothing():
    found = resolve("chorrent", KNOWN)
    assert found.project is None and not found.candidates


def test_mentioned_projects():
    assert [p.name for p in mentioned_in("Open dumbpipe in a new vs code window", KNOWN)] == ["dumbpipe"]
    assert [p.name for p in mentioned_in("how is dumb pipe doing?", KNOWN)] == ["dumbpipe"]
    assert mentioned_in("open notepad", KNOWN) == []


# ---------------------------------------------------------------------------
#  The index: SQLite + fact memory
# ---------------------------------------------------------------------------
def test_refresh_stores_projects_and_their_facts(tmp_path):
    _project(tmp_path, "dumbpipe")
    (tmp_path / "dumbpipe" / "README.md").write_text("# Dumb pipe\n\n[![CI](x)](y)\nConnect two machines with [iroh](z).\n")
    memory = FakeMemory()
    index = ProjectIndex(get_db(), memory, roots=[tmp_path], vscode_dir=tmp_path / "none")

    assert index.refresh() == 1
    [stored] = get_db().list_projects()
    assert stored.name == "dumbpipe" and stored.kind == "Rust"
    assert stored.description == "Connect two machines with iroh."
    assert "Project dumbpipe" in stored.summary and "Rust project" in stored.summary
    assert list(memory.facts.values()) == [stored.summary]


def test_unchanged_projects_are_not_re_embedded(tmp_path):
    _project(tmp_path, "sendme")
    memory = FakeMemory()
    index = ProjectIndex(get_db(), memory, roots=[tmp_path], vscode_dir=tmp_path / "none")
    index.refresh()
    index.refresh()
    assert memory.writes == 1


def test_the_users_choice_of_app_survives_a_refresh(tmp_path):
    folder = _project(tmp_path, "sendme")
    index = ProjectIndex(get_db(), FakeMemory(), roots=[tmp_path], vscode_dir=tmp_path / "none")
    index.refresh()
    get_db().set_project_opener(str(folder), "VS Code")
    index.refresh()
    [stored] = get_db().list_projects()
    assert stored.opened_with == "VS Code"
    assert "opens it with VS Code" in stored.summary


def test_a_deleted_project_is_forgotten(tmp_path):
    folder = _project(tmp_path, "old-thing")
    memory = FakeMemory()
    index = ProjectIndex(get_db(), memory, roots=[tmp_path], vscode_dir=tmp_path / "none")
    index.refresh()
    (folder / "Cargo.toml").unlink()
    folder.rmdir()
    index.refresh()
    assert get_db().list_projects() == []
    assert memory.facts == {}


def test_named_projects_reach_the_prompt(tmp_path):
    _project(tmp_path, "dumbpipe")
    ProjectIndex(get_db(), None, roots=[tmp_path], vscode_dir=tmp_path / "none").refresh()
    facts = project_facts(get_db(), "open dumbpipe in vs code", recalled=[])
    assert "Project dumbpipe" in facts
    block = build_live_context_block(get_db(), "", None, facts)
    assert "[KNOWN FACTS" in block and str(tmp_path / "dumbpipe") in block
    assert project_facts(get_db(), "open notepad", recalled=[]) == ""


# ---------------------------------------------------------------------------
#  Opening: the exact command line
# ---------------------------------------------------------------------------
@pytest.fixture
def launched(monkeypatch, tmp_path):
    """Captures what would be started instead of starting it."""
    argvs: list[list[str]] = []
    monkeypatch.setattr(desktop, "_start", argvs.append)
    monkeypatch.setattr(desktop, "_app_exe", lambda profile_id: f"C:/fake/{profile_id}.exe")
    monkeypatch.setattr(desktop, "vscode_last_active_folder", lambda: tmp_path / "sendme")
    monkeypatch.setattr(desktop, "get_project_index", lambda: None)
    for name in ("dumbpipe", "sendme"):
        folder = _project(tmp_path, name)
        get_db().upsert_project(Project(path=str(folder), name=name, summary=name, refreshed_at="x"))
    return argvs


def test_open_project_in_a_new_vs_code_window(launched, tmp_path):
    reply = desktop.launch_app("vs code", target="dumbpipe", window="new")
    assert launched == [["C:/fake/vscode.exe", "--new-window", str(tmp_path / "dumbpipe")]]
    assert "new VS Code window" in reply


def test_open_project_in_this_vs_code_window(launched, tmp_path):
    reply = desktop.launch_app("vscode", target="dumbpipe", window="this")
    assert launched == [["C:/fake/vscode.exe", "--reuse-window", str(tmp_path / "dumbpipe")]]
    assert "replacing sendme" in reply


def test_open_project_without_a_window_choice_uses_the_apps_default(launched, tmp_path):
    desktop.launch_app("code", target="dumb pipe")
    assert launched == [["C:/fake/vscode.exe", str(tmp_path / "dumbpipe")]]


def test_a_new_empty_vs_code_window(launched):
    reply = desktop.launch_app("vs code", window="new")
    assert launched == [["C:/fake/vscode.exe", "--new-window"]]
    assert "new, empty VS Code window" in reply


def test_opening_records_the_app_for_next_time(launched, tmp_path):
    desktop.launch_app("vs code", target="sendme")
    launched.clear()
    reply = desktop.launch_app("sendme")  # "open sendme": no app named
    assert launched == [["C:/fake/vscode.exe", str(tmp_path / "sendme")]]
    assert "last time" in reply


def test_a_project_with_no_known_app_gets_a_question(launched):
    reply = desktop.launch_app("dumbpipe")
    assert launched == []
    assert "Ask the user which app" in reply


def test_unknown_project_is_not_guessed(launched):
    reply = desktop.launch_app("vs code", target="chorrent")
    assert launched == []
    assert "don't know a project or folder called 'chorrent'" in reply


def test_an_app_that_cannot_open_folders_says_so(launched):
    reply = desktop.launch_app("notepad", target="dumbpipe")
    assert launched == []
    assert "cannot open a folder" in reply


def test_explorer_opens_a_project_folder(launched, tmp_path):
    desktop.launch_app("file explorer", target="dumbpipe")
    assert launched == [["C:/fake/explorer.exe", str(tmp_path / "dumbpipe")]]


# ---------------------------------------------------------------------------
#  Never-assume for the window choice
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "said,window,kept",
    [
        ("open a new instance of vs code please", "new", True),
        ("open another vs code window", "new", True),
        ("open dumbpipe in this vs code window", "this", True),
        ("open dumbpipe in the current vs code window", "this", True),
        ("open dumbpipe in vs code", "this", False),  # invented: dropped, default window
        ("open dumbpipe in vs code", "new", False),
    ],
)
def test_window_choice_needs_the_users_words(said, window, kept):
    check = check_call(get_tool("launch_app"), {"app_name": "vs code", "window": window}, said)
    assert check.ok
    assert ("window" in check.arguments) is kept
    assert ("window" in check.dropped) is not kept


def test_project_index_is_off_in_tests():
    assert projects.PROJECT_INDEX is False


def test_app_names_written_as_identifiers(launched, tmp_path):
    """Seen live: app_name="file_explorer"."""
    desktop.launch_app("file_explorer", target="sendme")
    assert launched == [["C:/fake/explorer.exe", str(tmp_path / "sendme")]]


def test_project_named_as_the_app_with_its_path_as_target(launched, tmp_path):
    """Seen live: launch_app(app_name="sendme", target=<sendme's folder>) — no app named."""
    reply = desktop.launch_app("sendme", target=str(tmp_path / "sendme"))
    assert launched == []
    assert "Ask the user which app" in reply
