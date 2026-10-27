"""Starting without being asked: shortcuts, engine recovery, and warm-up."""

from __future__ import annotations

import asyncio
import sys
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from gideon.brain import llm
from gideon.interfaces import shortcuts

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows autostart")


# ---------------------------------------------------------------------------
#  Shortcuts
# ---------------------------------------------------------------------------
@pytest.fixture
def fake_programs(tmp_path, monkeypatch):
    monkeypatch.setattr(shortcuts, "STARTUP_SHORTCUT", tmp_path / "Startup" / "G.I.D.E.O.N.lnk")
    monkeypatch.setattr(shortcuts, "START_MENU_SHORTCUT", tmp_path / "G.I.D.E.O.N.lnk")

    def fake_run(cmd, env, **kwargs):
        # Stand-in for PowerShell's WScript.Shell: just create the file.
        from pathlib import Path

        Path(env["GIDEON_LNK"]).write_text(env["GIDEON_TARGET"])
        fake_run.env = env
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(shortcuts.subprocess, "run", fake_run)
    return fake_run


def test_toggle_adds_and_removes_the_logon_shortcut(fake_programs):
    assert shortcuts.autostart_enabled() is False
    assert shortcuts.set_autostart(True) is True
    assert shortcuts.STARTUP_SHORTCUT.exists()
    assert shortcuts.set_autostart(False) is False
    assert not shortcuts.STARTUP_SHORTCUT.exists()


def test_shortcut_runs_the_tray_without_a_console(fake_programs):
    """python.exe would open a console window that kills the tray when closed."""
    shortcuts.set_autostart(True)
    assert fake_programs.env["GIDEON_TARGET"].lower().endswith("pythonw.exe")
    assert fake_programs.env["GIDEON_ARGS"].endswith("gideon_tray.py")


def test_toggle_and_installer_manage_the_same_file():
    source = (shortcuts.BASE_DIR / "install_autostart.ps1").read_text(encoding="utf-8")
    assert f"$ShortcutName = '{shortcuts.SHORTCUT_NAME}'" in source


def test_start_menu_entry_is_never_overwritten(fake_programs):
    shortcuts.START_MENU_SHORTCUT.write_text("edited by the user")
    shortcuts.ensure_start_menu_entry()
    assert shortcuts.START_MENU_SHORTCUT.read_text() == "edited by the user"


def test_paths_are_passed_by_environment_not_spliced_into_the_script():
    assert "$env:GIDEON_LNK" in shortcuts._CREATE_SHORTCUT_PS
    assert "{" not in shortcuts._CREATE_SHORTCUT_PS.replace("${", "")


# ---------------------------------------------------------------------------
#  Engine recovery
# ---------------------------------------------------------------------------
def _engine(tray, monkeypatch, health, process_running):
    engine = tray.Engine()
    probes = iter(health)
    monkeypatch.setattr(engine, "probe", lambda: next(probes, health[-1]))
    monkeypatch.setattr(tray, "engine_process_running", lambda: process_running)
    monkeypatch.setattr(tray, "WATCH_INTERVAL_SECONDS", 0)
    engine.start = MagicMock()
    return engine


def _watch(engine, polls):
    stop = threading.Event()
    calls = {"n": 0}
    real_wait = stop.wait

    def wait(timeout):
        calls["n"] += 1
        if calls["n"] > polls:
            stop.set()
        return real_wait(0)

    stop.wait = wait
    engine.watch(stop)


def test_dead_engine_is_restarted(monkeypatch):
    from gideon.interfaces import tray

    engine = _engine(tray, monkeypatch, health=[False, False, False], process_running=False)
    _watch(engine, polls=2)
    engine.start.assert_called_once()


def test_one_missed_probe_is_not_a_death(monkeypatch):
    from gideon.interfaces import tray

    engine = _engine(tray, monkeypatch, health=[False, True, True], process_running=False)
    _watch(engine, polls=3)
    engine.start.assert_not_called()


def test_a_loading_engine_is_not_duplicated(monkeypatch):
    """The boot task's engine takes a while to answer; a second one would
    fight it for the port and for Telegram's single polling slot."""
    from gideon.interfaces import tray

    engine = _engine(tray, monkeypatch, health=[False] * 5, process_running=True)
    _watch(engine, polls=4)
    engine.start.assert_not_called()


def test_engine_processes_are_recognised(monkeypatch):
    from gideon.interfaces import tray

    procs = [
        SimpleNamespace(info={"cmdline": ["pythonw.exe", "C:\\G\\gideon_tray.py"]}),
        SimpleNamespace(info={"cmdline": ["pythonw.exe", "C:\\G\\run_engine.py"]}),
    ]
    monkeypatch.setattr(tray.psutil, "process_iter", lambda attrs: iter(procs))
    assert tray.engine_process_running() is True

    monkeypatch.setattr(tray.psutil, "process_iter", lambda attrs: iter(procs[:1]))
    assert tray.engine_process_running() is False


# ---------------------------------------------------------------------------
#  Warm-up
# ---------------------------------------------------------------------------
def test_warm_up_is_skipped_on_battery(monkeypatch):
    monkeypatch.setattr(llm, "WARM_UP_MODEL", True)
    monkeypatch.setattr(llm, "on_battery", lambda: True)
    load = MagicMock()
    monkeypatch.setattr(llm, "_load_model", load)
    asyncio.run(llm.warm_up())
    load.assert_not_called()


def test_warm_up_waits_for_ollama(monkeypatch):
    """At boot Ollama is not running until someone logs in."""
    monkeypatch.setattr(llm, "WARM_UP_MODEL", True)
    monkeypatch.setattr(llm, "on_battery", lambda: False)
    monkeypatch.setattr(llm, "_WARM_UP_RETRY_SECONDS", (0,))
    load = MagicMock(side_effect=[ConnectionError("refused"), ConnectionError("refused"), None])
    monkeypatch.setattr(llm, "_load_model", load)
    asyncio.run(llm.warm_up())
    assert load.call_count == 3


def test_warm_up_prefills_the_real_prompt_prefix(monkeypatch):
    from gideon.brain.prompt import STATIC_SYSTEM_PROMPT
    from gideon.core.config import NUM_CTX

    fake = MagicMock()
    monkeypatch.setattr(llm, "ollama", fake)
    monkeypatch.setattr(llm, "on_battery", lambda: False)
    llm._load_model()
    kwargs = fake.chat.call_args.kwargs
    assert kwargs["messages"][0]["content"] == STATIC_SYSTEM_PROMPT
    assert kwargs["options"]["num_ctx"] == NUM_CTX


# ---------------------------------------------------------------------------
#  Log noise (2026-10-27: health polls and network-drop tracebacks were most
#  of a 5 MB engine log)
# ---------------------------------------------------------------------------
def _record(name, msg, args=(), exc=None):
    import logging

    exc_info = (type(exc), exc, None) if exc is not None else None
    return logging.LogRecord(name, logging.ERROR, __file__, 1, msg, args, exc_info)


def test_health_polls_are_not_logged():
    from gideon.core.logs import QuietHealthChecks

    quiet = QuietHealthChecks()
    assert not quiet.filter(_record("uvicorn.access", '%s - "%s %s HTTP/%s" %d', ("127.0.0.1:1", "GET", "/api/health", "1.1", 200)))
    assert quiet.filter(_record("uvicorn.access", '%s - "%s %s HTTP/%s" %d', ("127.0.0.1:1", "POST", "/api/chat", "1.1", 200)))


def test_a_network_drop_is_one_warning_line():
    import logging

    from gideon.core.logs import NetworkDropsAsOneLine

    network_error = type("NetworkError", (Exception,), {"__module__": "telegram.error"})
    timed_out = type("TimedOut", (network_error,), {"__module__": "telegram.error"})
    for exc in (network_error("httpx.ConnectError: getaddrinfo failed"), timed_out("Timed out")):
        record = _record("telegram.ext.Updater", "Exception happened while polling for updates.", exc=exc)
        assert NetworkDropsAsOneLine().filter(record)
        assert record.levelno == logging.WARNING and record.exc_info is None
        assert "retrying" in record.getMessage()


def test_other_telegram_errors_keep_their_traceback():
    import logging

    from gideon.core.logs import NetworkDropsAsOneLine

    record = _record("telegram.ext.Updater", "boom", exc=ValueError("a real bug"))
    assert NetworkDropsAsOneLine().filter(record)
    assert record.levelno == logging.ERROR and record.exc_info is not None
