"""
Unit tests for G.I.D.E.O.N Terminal Execution Tool and Endpoint.
Tests command execution, output capture, registry dispatch, and API endpoints.
"""

from unittest.mock import patch

from fastapi.testclient import TestClient

from gideon.core.config import PORT
from gideon.core.security import API_TOKEN, API_TOKEN_HEADER
from gideon.interfaces.api.app import app
from gideon.tools.registry import dispatch_tool_call, get_tool
from gideon.tools.terminal import run_terminal_command

# base_url sets the Host header; the server rejects anything non-loopback to
# defeat DNS rebinding, and TestClient's default "testserver" is not.
client = TestClient(app, base_url=f"http://localhost:{PORT}")
AUTH = {API_TOKEN_HEADER: API_TOKEN}


def test_run_terminal_command_success():
    res = run_terminal_command("echo Hello GIDEON Terminal")
    assert "Hello GIDEON Terminal" in res
    assert "Return Code: 0" in res


def test_run_terminal_command_empty():
    res = run_terminal_command("")
    assert "Error: No command specified" in res


def test_run_terminal_command_registry_dispatch():
    with patch.object(get_tool("run_terminal_command"), "fn", return_value="Registry Terminal OK") as fn:
        res = dispatch_tool_call("run_terminal_command", {"command": "git status"})
    assert res == "Registry Terminal OK"
    # The requesting channel is passed through for the audit trail.
    fn.assert_called_once_with(command="git status", cwd=None, channel="web")


def test_api_terminal_execute_endpoint():
    response = client.post(
        "/api/terminal/execute",
        json={"command": "echo API Terminal Test"},
        headers=AUTH,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "API Terminal Test" in data["output"]


def test_api_terminal_execute_empty_command():
    response = client.post(
        "/api/terminal/execute",
        json={"command": "   "},
        headers=AUTH,
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Command string cannot be empty"


# ---------------------------------------------------------------------------
#  Safety wiring
# ---------------------------------------------------------------------------
def test_privileged_tools_are_declared():
    """The shell always waits for approval; both it and app launching are
    privileged, so outside content can never reach them unattended."""
    shell, launcher = get_tool("run_terminal_command"), get_tool("launch_app")
    assert shell.confirm is True
    assert shell.privileged is True
    assert launcher.privileged is True


def test_drawer_commands_are_not_gated(monkeypatch):
    """The HUD's terminal drawer is the user typing the command themselves:
    that is the approval, so the tool function must not block it."""
    import gideon.tools.terminal as t

    result = t.run_terminal_command("echo DRAWER_RAN")
    assert "DRAWER_RAN" in result
    assert "Return Code: 0" in result


def test_terminal_publishes_threadsafe_not_direct():
    """Tools run in FastAPI's worker threads, not the event loop.
    asyncio.Queue.put_nowait is not thread-safe from another thread."""
    source = (
        __import__("pathlib").Path(__file__).parent.parent / "gideon" / "tools" / "terminal.py"
    ).read_text(encoding="utf-8")
    assert "bus.publish_threadsafe(" in source
    assert "bus.publish(" not in source


# ---------------------------------------------------------------------------
#  Endpoint authentication
# ---------------------------------------------------------------------------
def test_terminal_endpoint_rejects_missing_token():
    """Previously any local process could POST here and get a shell."""
    res = client.post("/api/terminal/execute", json={"command": "echo NOPE"})
    assert res.status_code == 401


def test_terminal_endpoint_rejects_wrong_token():
    res = client.post(
        "/api/terminal/execute",
        json={"command": "echo NOPE"},
        headers={API_TOKEN_HEADER: "not-the-token"},
    )
    assert res.status_code == 401


def test_terminal_audit_endpoint_requires_token():
    assert client.get("/api/terminal/audit").status_code == 401
    assert client.get("/api/terminal/audit", headers=AUTH).status_code == 200


def test_foreign_host_header_is_rejected():
    """DNS rebinding makes an attacker page same-origin; CORS cannot help,
    but the Host header still gives it away."""
    res = client.post(
        "/api/terminal/execute",
        json={"command": "echo NOPE"},
        headers={**AUTH, "Host": "evil.example.com"},
    )
    assert res.status_code == 400
    assert "Host" in res.json()["detail"]


def test_index_serves_the_token_to_the_page():
    res = client.get("/")
    assert res.status_code == 200
    assert "GIDEON_API_TOKEN" in res.text
