"""Start Menu launching, search roots, session titles, and notes/tasks."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from gideon.brain.titles import PLACEHOLDER_TITLE, SessionTitler, clean_title, fallback_title
from gideon.core.config import PORT
from gideon.storage import projects
from gideon.storage.db import get_db
from gideon.tools import desktop, notes

SHORTCUTS = {
    "google chrome": Path("C:/SM/Google Chrome.lnk"),
    "chrome remote desktop": Path("C:/SM/Chrome Remote Desktop.lnk"),
    "visual studio code": Path("C:/SM/Visual Studio Code.lnk"),
    "obs studio": Path("C:/SM/OBS Studio.lnk"),
    "word": Path("C:/SM/Word.lnk"),
    "wordpad": Path("C:/SM/WordPad.lnk"),
    "paint 3d": Path("C:/SM/Paint 3D.lnk"),
    "paint.net": Path("C:/SM/paint.net.lnk"),
}


# ---------------------------------------------------------------------------
#  Start Menu launching
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "name,expected",
    [
        ("word", "word"),                       # exact beats "wordpad"
        ("obs", "obs studio"),                  # unique partial
        ("chrome", "google chrome"),            # ends-with disambiguates
        ("visual studio code", "visual studio code"),
    ],
)
def test_shortcut_matching(name, expected):
    path, _ = desktop._match_shortcut(name, SHORTCUTS)
    assert path == SHORTCUTS[expected]


def test_ambiguous_name_asks_instead_of_guessing():
    path, candidates = desktop._match_shortcut("paint", SHORTCUTS)
    assert path is None
    assert set(candidates) == {"paint 3d", "paint.net"}


def test_installed_app_launches_from_its_shortcut(monkeypatch):
    monkeypatch.setattr(desktop, "installed_shortcuts", lambda: SHORTCUTS)
    with patch.object(desktop.os, "startfile", create=True) as startfile:
        result = desktop.launch_app("OBS")
    startfile.assert_called_once_with(str(SHORTCUTS["obs studio"]))
    assert "Launched" in result


def test_command_lines_never_reach_the_shortcut_matcher(monkeypatch):
    """Only a plain name is matched; anything shaped like a command is refused."""
    monkeypatch.setattr(desktop, "installed_shortcuts", lambda: SHORTCUTS)
    with patch.object(desktop.os, "startfile", create=True) as startfile:
        result = desktop.launch_app("word & calc")
    startfile.assert_not_called()
    assert "not on the approved application list" in result


def test_uninstallers_are_not_offered(tmp_path, monkeypatch):
    (tmp_path / "Foo").mkdir()
    for name in ("Foo.lnk", "Uninstall Foo.lnk", "Foo Help.lnk"):
        (tmp_path / "Foo" / name).write_bytes(b"")
    monkeypatch.setattr(desktop, "_START_MENU_DIRS", [tmp_path])
    monkeypatch.setattr(desktop, "_shortcuts_loaded_at", 0.0)
    monkeypatch.setattr(desktop, "_shortcuts", {})
    assert set(desktop.installed_shortcuts()) == {"foo"}


def test_search_roots_do_not_depend_on_the_working_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(projects, "SEARCH_ROOTS", [])
    assert tmp_path not in projects.default_search_roots()


def test_search_roots_can_be_configured(monkeypatch, tmp_path):
    monkeypatch.setattr(projects, "SEARCH_ROOTS", [str(tmp_path)])
    (tmp_path / "needle-report.pdf").write_text("x")
    assert "needle-report.pdf" in desktop.search_files("needle")


# ---------------------------------------------------------------------------
#  Session titles
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "raw,expected",
    [
        ('"GPU Temperature Check."', "GPU Temperature Check"),
        ("Title: Weekend plans\nmore text", "Weekend plans"),
        ("**Paris weather**", "Paris weather"),
    ],
)
def test_titles_are_cleaned(raw, expected):
    assert clean_title(raw) == expected


def test_fallback_title_uses_the_first_words():
    assert fallback_title("what's the weather in paris tomorrow morning please") == (
        "What's the weather in paris tomorrow"
    )


def _titler(reply="Paris weather check"):
    chat = MagicMock(return_value={"message": {"content": reply}})
    titler = SessionTitler(get_db(), publish=MagicMock(), chat=chat)
    # Run inline so the test does not race the worker thread.
    titler._executor = MagicMock(submit=lambda fn, *a: fn(*a))
    return titler, chat


def test_placeholder_sessions_get_titled_once():
    db = get_db()
    session = db.create_session(PLACEHOLDER_TITLE, auto_title=True)
    titler, chat = _titler()

    titler.maybe_title(session, "what's the weather in paris", "It is 20 degrees.")
    titler.maybe_title(session, "and tomorrow?", "Rain.")

    names = {s["session_id"]: s["session_name"] for s in db.list_sessions()}
    assert names[session] == "Paris weather check"
    assert chat.call_count == 1
    titler.publish.assert_called_once_with(
        "session_renamed", {"session_id": session, "session_name": "Paris weather check"}
    )


def test_named_sessions_are_left_alone():
    db = get_db()
    session = db.create_session("My own name")
    titler, chat = _titler()
    titler.maybe_title(session, "hello there", "Hi.")
    chat.assert_not_called()


def test_title_request_keeps_the_chat_models_context_size():
    """A different num_ctx makes Ollama reload the model."""
    from gideon.core.config import OLLAMA_OPTIONS

    session = get_db().create_session(PLACEHOLDER_TITLE, auto_title=True)
    titler, chat = _titler()
    titler.maybe_title(session, "hello there friend", "Hi.")
    assert chat.call_args.kwargs["options"]["num_ctx"] == OLLAMA_OPTIONS["num_ctx"]


def test_blank_session_name_is_auto_titled():
    from gideon.interfaces.api.app import app

    client = TestClient(app, base_url=f"http://localhost:{PORT}")
    body = client.post("/api/sessions", json={"name": ""}).json()
    assert body["session_name"] == PLACEHOLDER_TITLE
    assert get_db().session_wants_title(body["session_id"])

    named = client.post("/api/sessions", json={"name": "Taxes"}).json()
    assert not get_db().session_wants_title(named["session_id"])


# ---------------------------------------------------------------------------
#  Notes and tasks
# ---------------------------------------------------------------------------
def test_task_lifecycle():
    saved = notes.save_note("renew passport", kind="task")
    task_id = int(saved.split("#")[1].split(":")[0])

    assert "renew passport" in notes.list_notes(kind="tasks")
    assert "Marked task" in notes.complete_task(task_id)
    assert "renew passport" not in notes.list_notes(kind="task")
    assert "no open task" in notes.complete_task(task_id)


def test_notes_are_searchable():
    notes.save_note("wifi password is on the router sticker")
    assert "router sticker" in notes.list_notes(query="wifi")
    assert "No" in notes.list_notes(query="zzz-not-there")


def test_notes_are_available_over_telegram():
    from gideon.tools.registry import tools_for_channel

    names = {t["function"]["name"] for t in tools_for_channel("telegram")}
    assert {"save_note", "list_notes", "complete_task"} <= names


@pytest.mark.parametrize(
    "prompt",
    ["add buy milk to my to-do list", "what's on my to do list", "jot down the wifi password"],
)
def test_note_requests_are_offered_tools(prompt):
    """"to-do" splits into "to" + "do", so the tool's own trigger never saw it,
    tools were withheld, and the model claimed to have saved the item."""
    from gideon.brain.intent import should_offer_tools

    assert should_offer_tools(prompt)
