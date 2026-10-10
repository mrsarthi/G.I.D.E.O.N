"""Traces (PRD §6.3) and the eval runner that is built on them."""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from gideon.brain.agent import Agent
from gideon.brain.approvals import ApprovalQueue
from gideon.brain.traces import looks_like_correction, ollama_stats
from gideon.core.config import PORT
from gideon.core.security import API_TOKEN, API_TOKEN_HEADER
from gideon.evals.runner import Case, fake_tool, load_cases, score
from gideon.storage.db import get_db
from gideon.tools.registry import get_tool


def _chat(*turns):
    """Scripted ollama.chat: each turn is a reply string or a list of tool calls."""
    queue = list(turns)

    def chat(**kwargs):
        turn = queue.pop(0)
        done = SimpleNamespace(
            message=SimpleNamespace(content="", tool_calls=None),
            done=True,
            load_duration=5_000_000,
            prompt_eval_count=120,
            prompt_eval_duration=40_000_000,
            eval_count=20,
            eval_duration=500_000_000,
        )
        if isinstance(turn, str):
            return iter([{"message": {"content": turn}}, done])
        calls = [{"function": {"name": n, "arguments": a}} for n, a in turn]
        return iter([{"message": {"tool_calls": calls}}, done])

    return chat


def _run(agent: Agent, prompt: str):
    session = get_db().create_session("trace test")
    events: list[dict] = []
    agent.record_user_message(session, prompt)
    reply = agent.respond(session, prompt, on_event=events.append)
    trace_id = next(e["trace_id"] for e in events if "trace_id" in e)
    return session, reply, get_db().get_trace(trace_id)


# ---------------------------------------------------------------------------
#  What a trace records
# ---------------------------------------------------------------------------
def test_plain_answer_is_traced():
    agent = Agent(get_db(), None, chat=_chat("Hello there."))
    _, reply, trace = _run(agent, "hi")

    assert trace["outcome"] == "answered"
    assert trace["request"] == "hi"
    assert trace["reply"] == reply == "Hello there."
    assert trace["first_text_ms"] is not None
    assert trace["total_ms"] >= trace["first_text_ms"]
    assert trace["placement"] in ("gpu", "cpu")


def test_tool_turn_records_the_call_and_ollama_timings():
    agent = Agent(
        get_db(),
        None,
        chat=_chat([("get_weather", {"city": "Tokyo"})], "It is sunny."),
        tool_executor=fake_tool,
    )
    _, _, trace = _run(agent, "what's the weather in Tokyo?")

    assert trace["outcome"] == "tool"
    assert trace["tools_offered"] is True
    first_pass, second_pass = trace["detail"]["passes"]
    call = first_pass["tool_calls"][0]
    assert call["name"] == "get_weather"
    assert call["source"] == "native"
    assert call["args"] == {"city": "Tokyo"}
    assert first_pass["prompt_tokens"] == 120
    assert first_pass["tokens_per_sec"] == 40.0
    assert "history" in trace["detail"]["stages"] and "prompt" in trace["detail"]["stages"]


def test_media_reroutes_are_recorded():
    agent = Agent(
        get_db(),
        None,
        chat=_chat([("control_media", {"action": "play_pause"})], "Playing."),
        tool_executor=fake_tool,
    )
    _, _, trace = _run(agent, "play Loser by Tame Impala")
    call = trace["detail"]["passes"][0]["tool_calls"][0]
    assert call["name"] == "play_youtube_music"
    assert call["corrected_from"] == "control_media"


def test_approval_requests_are_traced_and_linked():
    agent = Agent(
        get_db(), None, chat=_chat([("run_terminal_command", {"command": "dir"})], "Waiting.")
    )
    with patch.object(get_tool("run_terminal_command"), "fn") as fn:
        _, _, trace = _run(agent, "run dir in the terminal")
    fn.assert_not_called()

    assert trace["outcome"] == "approval_requested"
    approval = trace["detail"]["passes"][0]["tool_calls"][0]["approval"]
    action = get_db().get_action(approval["action_id"])
    assert action.trace_id == trace["trace_id"]


def test_failed_turns_are_traced_too():
    def broken(**kwargs):
        raise ConnectionError("ollama is down")

    _, _, trace = _run(Agent(get_db(), None, chat=broken), "hello there")
    assert trace["outcome"] == "offline"
    assert "ollama is down" in trace["error"]


def test_eval_tool_executor_replaces_real_tools():
    agent = Agent(
        get_db(),
        None,
        chat=_chat([("launch_app", {"app_name": "notepad"})], "Opened."),
        tool_executor=fake_tool,
    )
    with patch.object(get_tool("launch_app"), "fn") as fn:
        _run(agent, "open notepad")
    fn.assert_not_called()


def test_ollama_stats_ignore_unfinished_chunks():
    assert ollama_stats({"message": {"content": "x"}}) == {}


# ---------------------------------------------------------------------------
#  Corrections
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "text,expected",
    [
        ("no, I meant Tokyo", True),
        ("Actually make it 5pm", True),
        ("that's wrong", True),
        ("I meant the other file", True),
        ("now play some music", False),
        ("nothing else, thanks", False),
        ("notepad please", False),
    ],
)
def test_correction_detection(text, expected):
    assert looks_like_correction(text) is expected


def test_a_correcting_message_is_recorded_on_the_previous_turn():
    agent = Agent(get_db(), None, chat=_chat("It's sunny in Osaka.", "Sorry — Tokyo is sunny."))
    session, _, first = _run(agent, "weather in Osaka?")

    agent.record_user_message(session, "no, I meant Tokyo")
    corrected = get_db().get_trace(first["trace_id"])
    assert corrected["correction_kind"] == "rephrase"
    assert corrected["correction_text"] == "no, I meant Tokyo"


def test_old_turns_are_not_corrected(monkeypatch):
    agent = Agent(get_db(), None, chat=_chat("Okay."))
    session, _, first = _run(agent, "hello")
    stale = (datetime.now() - timedelta(minutes=30)).isoformat(timespec="milliseconds")
    with get_db()._lock, get_db().conn:
        get_db().conn.execute(
            "UPDATE traces SET created_at = ? WHERE trace_id = ?", (stale, first["trace_id"])
        )

    agent.record_user_message(session, "no, that's wrong")
    assert get_db().get_trace(first["trace_id"])["correction_kind"] is None


def test_denied_approval_is_a_correction():
    agent = Agent(
        get_db(), None, chat=_chat([("run_terminal_command", {"command": "dir"})], "Waiting.")
    )
    _, _, trace = _run(agent, "run dir in the terminal")
    action_id = trace["detail"]["passes"][0]["tool_calls"][0]["approval"]["action_id"]

    ApprovalQueue(get_db()).deny(action_id)
    assert get_db().get_trace(trace["trace_id"])["correction_kind"] == "approval_denied"


# ---------------------------------------------------------------------------
#  Feedback API
# ---------------------------------------------------------------------------
def test_feedback_is_stored_on_the_trace():
    from gideon.interfaces.api.app import app

    _, _, trace = _run(Agent(get_db(), None, chat=_chat("Hi.")), "hello")
    client = TestClient(app, base_url=f"http://localhost:{PORT}")
    auth = {API_TOKEN_HEADER: API_TOKEN}

    assert client.post(f"/api/traces/{trace['trace_id']}/feedback", json={"rating": 1}).status_code == 401
    response = client.post(
        f"/api/traces/{trace['trace_id']}/feedback",
        json={"rating": -1, "note": "too long"},
        headers=auth,
    )
    assert response.status_code == 200
    stored = get_db().get_trace(trace["trace_id"])
    assert stored["rating"] == -1
    assert stored["correction_text"] == "too long"
    assert client.post("/api/traces/nope/feedback", json={"rating": 1}, headers=auth).status_code == 404


# ---------------------------------------------------------------------------
#  Eval scoring
# ---------------------------------------------------------------------------
def _trace(calls=(), reply="", tools_offered=True):
    return {
        "reply": reply,
        "tools_offered": tools_offered,
        "detail": {
            "passes": [
                {"tool_calls": [{"name": n, "args": a, "source": "native"} for n, a in calls]}
            ]
        },
    }


def test_eval_cases_file_is_valid():
    cases = load_cases()
    assert len(cases) >= 90
    tool_names = {"none"} | {t.name for t in __import__("gideon.tools.registry", fromlist=["x"]).all_tools()}
    for case in cases:
        assert set(case.tools) <= tool_names, case.id
        assert set(case.forbid) <= tool_names, case.id
        assert case.tools or case.ask, case.id


def test_scoring_tool_and_slots():
    case = Case("w", "weather in Tokyo", "slots", tools=["get_weather"], args={"location": "tokyo"})
    assert score(case, _trace([("get_weather", {"location": "Tokyo, Japan"})])) == (True, "ok")
    assert score(case, _trace([("get_weather", {"location": "Osaka"})])) == (False, "slot:location")
    assert score(case, _trace([("search_web", {})])) == (False, "wrong_tool:search_web")
    assert score(case, _trace([], tools_offered=False)) == (False, "gated")
    assert score(case, _trace([])) == (False, "no_tool")


def test_scoring_chat_and_clarify():
    chat = Case("c", "hi", "chat", tools=["none"])
    assert score(chat, _trace())[0] is True
    assert score(chat, _trace([("set_clipboard", {})])) == (False, "unexpected:set_clipboard")

    ask = Case("a", "remind me", "clarify", ask=True, forbid=["set_reminder"])
    assert score(ask, _trace(reply="When should I remind you?")) == (True, "asked")
    assert score(ask, _trace([("set_reminder", {})], "Done.")) == (False, "guessed:set_reminder")
    assert score(ask, _trace(reply="Sure thing.")) == (False, "no_question")
