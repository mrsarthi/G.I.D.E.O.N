"""Stopping the engine without killing it.

The tray restarts the engine routinely (tray menu, boot task), and a hard kill
skips lifespan teardown — which leaves Telegram's getUpdates consumer
registered, so the *next* start fails with 409 Conflict. Signals cannot solve
this on Windows (see lifecycle.py), so shutdown is an authenticated loopback
request, and these tests pin that it is both reachable and gated.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from gideon.core import lifecycle
from gideon.core.config import PORT
from gideon.core.security import API_TOKEN, API_TOKEN_HEADER
from gideon.interfaces.api.app import app

# TestClient's default "testserver" Host is rejected by the DNS-rebinding
# middleware.
client = TestClient(app, base_url=f"http://localhost:{PORT}")


@pytest.fixture(autouse=True)
def _unbind():
    """Never leave a real server bound across tests."""
    yield
    lifecycle.bind(None)


def test_shutdown_requires_the_api_token():
    """Any local process could otherwise stop the engine at will."""
    assert client.post("/api/shutdown").status_code == 401


def test_shutdown_rejects_a_wrong_token():
    response = client.post("/api/shutdown", headers={API_TOKEN_HEADER: "not-the-token"})
    assert response.status_code == 401


def test_shutdown_asks_uvicorn_to_stop():
    server = MagicMock()
    server.should_exit = False
    lifecycle.bind(server)

    response = client.post("/api/shutdown", headers={API_TOKEN_HEADER: API_TOKEN})

    assert response.status_code == 200
    assert response.json()["status"] == "stopping"
    assert server.should_exit is True


def test_shutdown_reports_when_no_server_is_bound():
    """run.bat and a bare `uvicorn` invocation do not bind one, and the caller
    needs to know so it can fall back to terminating the process."""
    lifecycle.bind(None)
    response = client.post("/api/shutdown", headers={API_TOKEN_HEADER: API_TOKEN})
    assert response.status_code == 503


def test_request_shutdown_is_false_without_a_server():
    lifecycle.bind(None)
    assert lifecycle.is_bound() is False
    assert lifecycle.request_shutdown() is False


def test_binding_makes_shutdown_available():
    lifecycle.bind(MagicMock())
    assert lifecycle.is_bound() is True
    assert lifecycle.request_shutdown() is True


# ---------------------------------------------------------------------------
#  The launcher
# ---------------------------------------------------------------------------
def test_launcher_binds_streams_before_importing_uvicorn():
    """pythonw.exe has no console, so sys.stderr is None and uvicorn dies on
    its first log line. The fix only works if it happens first."""
    source = (__import__("pathlib").Path(__file__).parent.parent / "run_engine.py").read_text(
        encoding="utf-8"
    )
    body = source.split("def main(")[1]
    assert body.index("bind_headless_streams(") < body.index("import uvicorn")


def test_tray_launcher_binds_streams_before_importing_anything():
    """Same trap, plus one more: importing pywebview swaps a missing stdout
    for a writer to `nul`, after which binding thinks a console exists. The
    tray therefore produced no log at all until the launcher bound first."""
    source = (
        __import__("pathlib").Path(__file__).parent.parent / "gideon_tray.py"
    ).read_text(encoding="utf-8")
    assert source.index("bind_headless_streams(") < source.index("from gideon.interfaces.tray")


def test_a_nul_stream_counts_as_no_console(tmp_path, monkeypatch):
    import io
    import logging
    import sys

    from gideon.core import logs

    nul = io.StringIO()
    nul.name = "nul"
    monkeypatch.setattr(sys, "stdout", nul)
    monkeypatch.setattr(sys, "stderr", nul)
    monkeypatch.setattr(logs, "LOG_DIR", tmp_path)
    stale = logging.StreamHandler(nul)
    logging.getLogger().addHandler(stale)
    try:
        path = logs.bind_headless_streams("probe")
        assert path == tmp_path / "probe.log"
        # A handler created before binding is re-pointed, not left writing to nul.
        assert stale.stream is sys.stdout
    finally:
        logging.getLogger().removeHandler(stale)
        sys.stdout.close()


def test_binding_is_a_noop_with_a_real_console():
    """Running from a terminal must keep printing to the terminal."""
    from gideon.core.logs import bind_headless_streams

    assert bind_headless_streams("unit-test-should-not-be-created") is None


def test_launcher_does_not_enable_reload():
    """A file watcher in a boot service restarts on any stray edit, which
    means dropping reminders."""
    source = (__import__("pathlib").Path(__file__).parent.parent / "run_engine.py").read_text(
        encoding="utf-8"
    )
    assert "reload=False" in source
    assert "reload=True" not in source
