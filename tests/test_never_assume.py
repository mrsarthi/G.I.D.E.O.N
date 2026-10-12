"""Never assume (PRD §5.5) and the claimed-action check (F-12).

Measured before these existed: llama3.2 asked a clarifying question 0 times
out of 10 — it always guessed — and qwen models answered "I will check your
GPU" without calling any tool.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from gideon.brain.agent import CLAIMED_ACTION_NUDGE, Agent, OpeningGate
from gideon.brain.intent import claims_action
from gideon.evals.runner import Case, fake_tool, score
from gideon.storage.db import get_db
from gideon.tools.registry import get_tool
from gideon.tools.reminders import set_reminder
from gideon.tools.validation import check_call, is_grounded


# ---------------------------------------------------------------------------
#  Grounding
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "value,said,expected",
    [
        ("in 30 minutes", "Remind me to drink water in 30 minutes", True),
        ("tomorrow at 9am", "set a reminder for tomorrow at 9am to call the bank", True),
        ("in 1 day", "remind me to call mom", False),  # invented time
        ("Starboy by The Weeknd", "put on some Arctic Monkeys", False),  # invented song
        ("last Formula One race results", "look up who won the last F1 race", True),  # paraphrase
        ("reminders", "show me my reminder", True),  # plural folding
        ("the file", "find the file", True),  # filler only: nothing to have invented
        ("calc", "open calculator", True),  # model's shortening of the user's word
        ("spotify", "open spo", False),  # too short to be evidence
        ("spotify", "launch spoti", True),  # user's shortening
        ("chrome", "open calculator", False),
        ("vscode", "launch vs code for me", True),  # spoken words run together
        ("vs code", "open chorrent in another vscode window", True),  # ...and the other way
    ],
)
def test_grounding(value, said, expected):
    assert is_grounded(value, said) is expected


def test_ignored_words_do_not_count_as_evidence():
    """"set a reminder for 6pm" contains the word reminder — that must not let
    a reminder text of "reminder" pass as something the user said."""
    assert is_grounded("reminder", "set a reminder for 6pm", ignore=("reminder",)) is True
    assert is_grounded("call the dentist", "set a reminder for 6pm", ignore=("reminder",)) is False


# ---------------------------------------------------------------------------
#  The broker's check
# ---------------------------------------------------------------------------
def test_missing_required_value_blocks_the_call():
    check = check_call(get_tool("set_reminder"), {"text": "call mom"}, "remind me to call mom")
    assert not check.ok
    assert check.problem() == "missing:time_str"
    assert "Ask the user" in check.instruction()


def test_a_pointing_word_is_not_a_value():
    """Eval 2026-10-12: "copy that to my clipboard" -> set_clipboard(text="That")."""
    check = check_call(get_tool("set_clipboard"), {"text": "That"}, "copy that to my clipboard")
    assert check.problem() == "missing:text"


def test_invented_required_value_blocks_the_call():
    check = check_call(
        get_tool("set_reminder"),
        {"text": "call mom", "time_str": "in 1 day"},
        "remind me to call mom",
    )
    assert check.problem() == "ungrounded:time_str"


def test_invented_optional_value_is_dropped_not_blocked():
    """An invented city for the weather falls back to the current location."""
    check = check_call(get_tool("get_weather"), {"city": "London"}, "is it going to rain today?")
    assert check.ok
    assert check.dropped == ["location"]
    assert "city" not in check.arguments


def test_grounded_values_pass():
    check = check_call(
        get_tool("set_reminder"),
        {"text": "drink water", "time_str": "in 30 minutes"},
        "Remind me to drink water in 30 minutes",
    )
    assert check.ok and not check.dropped


def test_reminder_no_longer_defaults_its_time():
    assert get_tool("set_reminder").params[1].default == ""
    assert "could not work out when" in set_reminder("call mom", "whenever-ish")


# ---------------------------------------------------------------------------
#  The agent: blocked calls become questions
# ---------------------------------------------------------------------------
def _chat(*turns):
    queue = list(turns)

    def chat(**kwargs):
        # Snapshot: the agent keeps appending to the same list afterwards.
        chat.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        turn = queue.pop(0)
        if kwargs.get("format"):
            # The constrained retry: not streamed, answers with JSON.
            assert isinstance(turn, dict), "expected a forced-decision turn"
            return {"message": {"content": json.dumps(turn)}}
        if isinstance(turn, str):
            return iter([{"message": {"content": turn}}])
        if isinstance(turn, tuple):  # streamed in pieces
            return iter([{"message": {"content": piece}} for piece in turn])
        return iter(
            [{"message": {"tool_calls": [{"function": {"name": n, "arguments": a}} for n, a in turn]}}]
        )

    chat.calls = []
    return chat


def _run(agent, prompt):
    session = get_db().create_session("never assume")
    events: list[dict] = []
    agent.record_user_message(session, prompt)
    reply = agent.respond(session, prompt, on_event=events.append)
    trace = get_db().get_trace(next(e["trace_id"] for e in events if "trace_id" in e))
    return reply, trace


def test_guessed_reminder_time_is_never_saved():
    chat = _chat(
        [("set_reminder", {"text": "call mom", "time_str": "in 1 day"})],
        "When would you like me to remind you?",
    )
    with patch.object(get_tool("set_reminder"), "fn") as fn:
        reply, trace = _run(Agent(get_db(), None, chat=chat, tool_executor=fake_tool), "remind me to call mom")

    fn.assert_not_called()
    call = trace["detail"]["passes"][0]["tool_calls"][0]
    assert call["blocked"] == "ungrounded:time_str"
    assert trace["outcome"] == "clarified"
    assert reply.endswith("?")
    # The model was told to ask, not given a fake success.
    tool_messages = [m for m in chat.calls[1]["messages"] if m["role"] == "tool"]
    assert "Ask the user" in tool_messages[-1]["content"]


# ---------------------------------------------------------------------------
#  Claimed actions
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "reply",
    [
        "Yes boss, I will check your GPU temperature right away.",
        "I have set a reminder for you to call the bank.",
        "Yes sir, I'll set that reminder for you.",
        "Checking your RAM now.",
        # Reported results with no tool call (seen live on qwen3:4b-instruct)
        "Yes boss. Notepad is now open. What would you like to write, sir?",
        "Yes boss. The calculator is now open.",
        "Your reminder has been set for 5 pm.",
        "Calculator launched successfully.",
        "The volume is now at 30 percent.",
        "Yes boss, I can check your GPU temperature right now. Let me retrieve that for you.",
        "I don't have access to real-time weather data. Try a weather app.",
    ],
)
def test_claims_are_detected(reply):
    assert claims_action(reply)


@pytest.mark.parametrize(
    "reply",
    [
        "Why is the sky blue? Rayleigh scattering.",
        "Let me explain how a hash map works.",
        "I am doing well, thanks for asking.",
        "Your GPU is at 52 degrees.",
        "The shop is open until 9 pm on weekdays.",
        "Python has been around since 1991.",
        "That setting is now deprecated in newer versions.",
        "I can set reminders, open apps and check the weather for you.",
    ],
)
def test_ordinary_replies_are_not_claims(reply):
    assert not claims_action(reply)


def test_a_claim_without_a_tool_forces_a_real_call():
    chat = _chat(
        "Yes boss, I will check your RAM usage for you.",  # the F-12 failure
        {"tool": "get_system_status", "arguments": {}},     # constrained retry
        "RAM is at 45%.",
    )
    reply, trace = _run(Agent(get_db(), None, chat=chat, tool_executor=fake_tool), "check my ram usage")

    names = [(c["name"], c["source"]) for p in trace["detail"]["passes"] for c in p["tool_calls"]]
    assert names == [("get_system_status", "forced")]
    assert trace["outcome"] == "tool"
    # The retry is schema-constrained: one offered tool or ask_user, nothing else.
    retry = chat.calls[1]
    assert retry["stream"] is False
    assert "ask_user" in retry["format"]["properties"]["tool"]["enum"]
    assert CLAIMED_ACTION_NUDGE in retry["messages"][-1]["content"]


def test_a_forced_call_still_goes_through_validation():
    """Forcing a decision must not let an invented value through."""
    chat = _chat(
        "Sure, I'll set that reminder.",
        {"tool": "set_reminder", "arguments": {"text": "call mom", "time_str": "in 1 day"}},
        "When should I remind you?",
    )
    with patch.object(get_tool("set_reminder"), "fn") as fn:
        _, trace = _run(Agent(get_db(), None, chat=chat, tool_executor=fake_tool), "remind me to call mom")
    fn.assert_not_called()
    assert trace["detail"]["passes"][0]["tool_calls"][0]["blocked"] == "ungrounded:time_str"


def test_a_forced_question_is_asked():
    chat = _chat("I'll set that up for you.", {"tool": "ask_user", "arguments": {}, "question": "For what time?"})
    reply, trace = _run(Agent(get_db(), None, chat=chat), "set a reminder please")
    # The claim was held back, so the question replaces it rather than following it.
    assert reply == "For what time?"
    assert trace["outcome"] == "clarified"


def test_a_claim_with_tools_withheld_still_gets_a_decision():
    """The "add buy milk to my to-do list" gating miss: the model meant to act."""
    chat = _chat(
        "Sure, I'll add that for you.",
        {"tool": "save_note", "arguments": {"text": "buy milk", "kind": "task"}},
        "Added.",
    )
    with patch("gideon.brain.agent.should_offer_tools", return_value=False):
        _, trace = _run(Agent(get_db(), None, chat=chat, tool_executor=fake_tool), "buy milk, on the list please")

    assert chat.calls[0]["tools"] == []
    assert chat.calls[1]["format"] is not None
    assert chat.calls[2]["tools"], "the follow-up pass must have tools"
    assert trace["tools_offered"] is True


def test_an_unusable_decision_leaves_the_reply_alone():
    chat = _chat("I'll check that.", {"tool": "teleport", "arguments": {}})
    reply, trace = _run(Agent(get_db(), None, chat=chat), "check my ram usage")
    assert reply == "I'll check that."
    assert len(chat.calls) == 2
    assert trace["outcome"] == "answered"


# ---------------------------------------------------------------------------
#  Scoring
# ---------------------------------------------------------------------------
def _trace(calls, reply=""):
    return {
        "reply": reply,
        "tools_offered": True,
        "detail": {"passes": [{"tool_calls": calls}]},
    }


def test_blocked_guess_counts_as_asking():
    case = Case("a", "remind me to call mom", "clarify", ask=True, forbid=["set_reminder"])
    blocked = [{"name": "set_reminder", "args": {}, "blocked": "ungrounded:time_str"}]
    assert score(case, _trace(blocked, "When should I remind you?")) == (True, "asked")


def test_blocked_correct_tool_counts_as_a_false_block():
    case = Case("r", "remind me in 30 minutes", "slots", tools=["set_reminder"])
    blocked = [{"name": "set_reminder", "args": {}, "blocked": "ungrounded:text"}]
    assert score(case, _trace(blocked)) == (False, "blocked:ungrounded:text")


def test_scorer_judges_the_value_the_tool_receives():
    case = Case("m", "pause the music", "media", tools=["control_media"], args={"action": "play_pause"})
    calls = [{"name": "control_media", "args": {"action": "pause"}}]
    assert score(case, _trace(calls)) == (True, "ok")


# ---------------------------------------------------------------------------
#  The opening gate: a false claim is replaced, not spoken
# ---------------------------------------------------------------------------
def test_a_held_claim_is_replaced_by_the_real_result():
    """Seen live: "Yes boss. Notepad is now open." with no tool call."""
    chat = _chat(
        ("Yes boss. ", "Notepad is ", "now open. What would you like to write?"),
        {"tool": "launch_app", "arguments": {"app_name": "notepad"}},
        "Notepad is open, sir.",
    )
    agent = Agent(get_db(), None, chat=chat, tool_executor=fake_tool)
    session = get_db().create_session("gate")
    agent.record_user_message(session, "open notepad")
    pieces = list(agent.stream(session, "open notepad"))

    assert "".join(pieces) == "Notepad is open, sir."
    # The next pass never saw the false claim as something it said.
    assert all("now open" not in (m.get("content") or "") for m in chat.calls[2]["messages"] if m["role"] == "assistant")


def test_invented_weather_is_replaced():
    chat = _chat(
        "The current weather in Varanasi is mostly cloudy with a high of 28°C.",
        {"tool": "get_weather", "arguments": {"location": "Varanasi"}},
        "It is clear and 24°C in Varanasi.",
    )
    reply, trace = _run(
        Agent(get_db(), None, chat=chat, tool_executor=fake_tool), "what's the weather like in varanasi today?"
    )
    assert reply == "It is clear and 24°C in Varanasi."
    assert trace["detail"]["passes"][0]["tool_calls"][0]["source"] == "forced"


def test_an_honest_reply_streams_once_its_opening_is_checked():
    long_first = "A light-year is the distance light travels in one year, about 9.46 trillion kilometres. "
    gate = OpeningGate(active=True)
    assert gate.feed("A light-year is ") == ""            # held while too short to judge
    released = gate.feed(long_first[len("A light-year is "):])
    assert released == long_first                         # whole first sentence, at once
    assert gate.feed("Proxima is 4.2 of them away.") == "Proxima is 4.2 of them away."


def test_the_gate_is_off_when_no_tools_were_offered():
    assert OpeningGate(active=False).feed("Yes boss. Notepad is now open.") == "Yes boss. Notepad is now open."


def test_the_models_own_sentence_is_not_the_users_text():
    """Eval 2026-10-12: "copy that to my clipboard" -> the model copied its own reply."""
    check = check_call(
        get_tool("set_clipboard"),
        {"text": "Yes boss, I'll copy that to your clipboard right away."},
        "copy that to my clipboard",
    )
    assert check.problem() == "ungrounded:text"
