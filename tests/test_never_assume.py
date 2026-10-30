"""Never assume (PRD §5.5) and the claimed-action check (F-12).

Measured before these existed: llama3.2 asked a clarifying question 0 times
out of 10 — it always guessed — and qwen models answered "I will check your
GPU" without calling any tool.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from gideon.brain.agent import CLAIMED_ACTION_NUDGE, CLARIFY_REPLY, Agent, OpeningGate
from gideon.brain.intent import NOT_DONE_REPLY, claims_action, is_command, is_decline, strip_claims
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
        "Certainly sir. I've triggered the media control to skip to the next track.",
        # A reading with no tool run is invented (the check only runs before a
        # tool has acted; eval 2026-10-30: "your battery is at 87%").
        "Your GPU is at 52 degrees.",
        "Yes boss, your battery is at 87%.",
        "Yes boss, here are the top trending stories from Hacker News:",
        # A bare past tense opening a sentence (2026-10-27, no tool call)
        "Paused the music, sir.",
        "Done. Opened Notepad for you.",
        "Launched Calculator.",
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
        "The shop is open until 9 pm on weekdays.",
        "Python has been around since 1991.",
        "That setting is now deprecated in newer versions.",
        "I can set reminders, open apps and check the weather for you.",
        "Set theory is a branch of mathematics.",
        "Opened in 1990, the museum is famous for its murals.",
        "It started raining yesterday.",
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
    forced = chat.calls[1]
    assert forced["format"] is not None
    # No schemas in the prompt, so the choices are described in the request.
    assert "save_note(" in forced["messages"][-1]["content"]
    # One tool list per turn (F-20): the reply pass reuses the cached prompt.
    assert chat.calls[2]["tools"] == []
    names = [(c["name"], c["source"]) for p in trace["detail"]["passes"] for c in p["tool_calls"]]
    assert names == [("save_note", "forced")]


def test_the_forced_decision_reuses_the_cached_prompt():
    """F-20: the "7-second penalty" was Ollama re-reading the conversation
    because the retry dropped the tool list and the reply pass narrowed it.
    Every call in the turn now sends the same tools and the same message
    prefix, so only the appended tokens are new."""
    chat = _chat(
        "Yes boss. Notepad is now open.",
        {"tool": "launch_app", "arguments": {"app_name": "notepad"}},
        "Notepad is open, sir.",
    )
    reply, trace = _run(Agent(get_db(), None, chat=chat, tool_executor=fake_tool), "open notepad")

    first, forced, follow_up = chat.calls
    assert first["tools"], "tools were offered for this request"
    assert forced["tools"] == first["tools"]
    assert follow_up["tools"] == first["tools"]
    # The retry extends the first pass's messages instead of rebuilding them.
    assert forced["messages"][: len(first["messages"])] == first["messages"]
    # Its choices are still limited, by the schema rather than the tool list.
    assert "launch_app" in forced["format"]["properties"]["tool"]["enum"]
    # And it is traced as a pass of its own, so its cost is visible.
    forced_pass = trace["detail"]["passes"][1]
    assert "launch_app" in forced_pass["forced_candidates"]
    assert reply == "Notepad is open, sir."


def test_an_unusable_decision_never_lets_the_claim_through():
    chat = _chat("I'll check that.", {"tool": "teleport", "arguments": {}})
    reply, trace = _run(Agent(get_db(), None, chat=chat), "check my ram usage")
    # Nothing replaced the promise, and nothing was checked: it is not spoken.
    assert reply == NOT_DONE_REPLY
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
        "The current weather in Lisbon is mostly cloudy with a high of 28°C.",
        {"tool": "get_weather", "arguments": {"location": "Lisbon"}},
        "It is clear and 24°C in Lisbon.",
    )
    reply, trace = _run(
        Agent(get_db(), None, chat=chat, tool_executor=fake_tool), "what's the weather like in lisbon today?"
    )
    assert reply == "It is clear and 24°C in Lisbon."
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



def test_strip_claims_keeps_the_question():
    reply = (
        'Yes sir. The "sendme" folder is now open in Visual Studio Code. Ready for your work, boss.\n\n'
        "Which app should I use, sir?"
    )
    assert strip_claims(reply) == "Which app should I use, sir?"
    assert strip_claims("Yes boss. Notepad is now open.") == NOT_DONE_REPLY


def test_a_claim_after_a_refused_tool_is_cut():
    """Seen live: a forced call was blocked, then the next pass said
    "The sendme folder is now open in Visual Studio Code"."""
    chat = _chat(
        "Opening it now.",
        {"tool": "set_clipboard", "arguments": {"folder_path": "sendme"}},
        'Yes sir. The "sendme" folder is now open in Visual Studio Code. What should I copy?',
    )
    reply, trace = _run(Agent(get_db(), None, chat=chat, tool_executor=fake_tool), "vs code please")
    assert "now open" not in reply
    assert reply == "What should I copy?"
    assert "claim_cut" in trace["detail"]["stages"]


def test_a_refusal_does_not_count_as_done():
    """launch_app answered "Not opened: …" — a later "is now open" is still false."""
    chat = _chat(
        [("launch_app", {"app_name": "notepad"})],
        "Notepad is now open, sir.",
    )

    def refusing(name, args, channel):
        return "Not opened: 'notepad' is approved but I couldn't find it installed on this system."

    reply, _ = _run(Agent(get_db(), None, chat=chat, tool_executor=refusing), "open notepad")
    assert reply == NOT_DONE_REPLY


# ---------------------------------------------------------------------------
#  After a command, any reported state is a claim (2026-10-27: "Calculator is
#  open, sir." / "Music paused, sir." with no tool call, six of six uncaught)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "reply",
    ["Calculator is open, sir.", "Music paused, sir.", "Music playback paused, sir.", "All set, boss.", "Done!"],
)
def test_reported_states_are_claims_after_a_command(reply):
    assert claims_action(reply, command=True)
    # The same words in conversation are not claims ("the shop is open").
    assert not claims_action(reply, command=False)


@pytest.mark.parametrize("reply", ["Which app should I open, sir?", "What time should I set it for?"])
def test_questions_after_a_command_are_not_claims(reply):
    assert not claims_action(reply, command=True)


def test_an_uncaught_state_claim_now_forces_a_real_call():
    chat = _chat(
        "Calculator is open, sir.",
        {"tool": "launch_app", "arguments": {"app_name": "calculator"}},
        "Calculator is open, sir.",
    )
    reply, trace = _run(Agent(get_db(), None, chat=chat, tool_executor=fake_tool), "open calculator")
    names = [(c["name"], c["source"]) for p in trace["detail"]["passes"] for c in p["tool_calls"]]
    assert names == [("launch_app", "forced")]
    assert reply == "Calculator is open, sir."  # now true: the tool ran


# ---------------------------------------------------------------------------
#  Eval 2026-10-27: three regressions from the changes above
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "message,expected",
    [
        ("open calculator", True),
        ("pause the music", True),
        ("Can you open a new instance of vs code please", True),
        ("hey gideon, set a reminder for 5", True),
        ("put on some Arctic Monkeys", True),
        ("review the code I just copied", True),
        ("I want to start running in the mornings, any tips?", False),
        ("what is the weather in oslo?", False),
        ("Vs code please", False),
    ],
)
def test_commands_are_imperative_requests(message, expected):
    assert is_command(message) is expected


def test_advice_is_not_a_claim_when_the_question_merely_mentions_an_action():
    """tricky-run-chat: tools were offered for "start" and "running"; the tips
    were taken for a claim and replaced by a reminder question."""
    tips = "Start slow, sir: run three easy mornings a week and keep a set bedtime. You'll be running daily soon."
    chat = _chat(tips)
    reply, trace = _run(Agent(get_db(), None, chat=chat), "I want to start running in the mornings, any tips?")
    assert reply == tips
    assert len(chat.calls) == 1, "no forced decision"


def test_the_retry_chooses_among_the_tools_the_model_already_had():
    """history-tools-pause: re-routing on the previous (octopus) exchange
    turned "pause the music" into a web search."""
    chat = _chat(
        "Music paused, sir.",
        {"tool": "control_media", "arguments": {"action": "play_pause"}},
        "Paused, sir.",
    )
    _run(Agent(get_db(), None, chat=chat, tool_executor=fake_tool), "pause the music")
    first, forced, _ = chat.calls
    offered = {t["function"]["name"] for t in first["tools"]}
    assert set(forced["format"]["properties"]["tool"]["enum"]) == offered | {"ask_user"}


def test_a_premature_claim_beside_a_real_call_is_dropped_quietly():
    """media-play-artist: "I haven't done that yet — it didn't go through." was
    glued to the real answer, though a tool call came with the claim."""
    calls = []

    def chat(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return iter([
                {"message": {"content": "Playing Arctic Monkeys now, boss. "}},
                {"message": {"tool_calls": [{"function": {"name": "play_youtube_music", "arguments": {"query": "Arctic Monkeys"}}}]}},
            ])
        return iter([{"message": {"content": "Arctic Monkeys are playing, boss."}}])

    reply, _ = _run(Agent(get_db(), None, chat=chat, tool_executor=fake_tool), "put on some Arctic Monkeys")
    assert reply == "Arctic Monkeys are playing, boss."
    # The next pass doesn't see the premature claim as something it said.
    assert all("Playing Arctic Monkeys now" not in str(m.get("content")) for m in calls[1]["messages"])



def test_the_request_repeated_is_not_a_value():
    """Eval 2026-10-27: set_clipboard(text="Copy that to your clipboard") ran."""
    check = check_call(get_tool("set_clipboard"), {"text": "Copy that to your clipboard"}, "copy that to my clipboard")
    assert check.problem() == "missing:text"


def test_a_generic_query_can_still_be_the_query():
    """search_web's "news" and "latest" are poor evidence but real content."""
    check = check_call(get_tool("search_web"), {"query": "latest news"}, "what's the latest news?")
    assert check.ok


def test_ask_user_without_a_question_still_asks():
    chat = _chat("I'll copy that for you.", {"tool": "ask_user", "arguments": {}, "question": ""})
    reply, trace = _run(Agent(get_db(), None, chat=chat), "copy that to my clipboard")
    assert reply == CLARIFY_REPLY
    assert trace["outcome"] == "clarified"


@pytest.mark.parametrize(
    "advice",
    [
        "Running in the mornings is a great habit, sir.",
        "Reading before bed helps you sleep.",
        "2. **Set a consistent time** - pick a time that works for you.",
        "Setting goals is important.",
    ],
)
def test_advice_is_not_a_claim(advice):
    """Eval 2026-10-27, tricky-run-chat: running tips were taken for claims."""
    assert not claims_action(advice)


def test_a_long_answer_is_checked_at_its_opening_only():
    tips = (
        "Yes sir, starting a morning running routine is a great way to build discipline and improve health. "
        "Here are a few practical tips:\n\n1. **Start slow** - begin with a short walk.\n\n"
        "2. **Set a consistent time** - pick a time that works for your schedule."
    )
    chat = _chat(tips)
    reply, _ = _run(Agent(get_db(), None, chat=chat), "I want to start running in the mornings, any tips?")
    assert reply == tips
    assert len(chat.calls) == 1, "no forced decision"


@pytest.mark.parametrize(
    "reply",
    [
        "Yes boss. Good morning to you. I'm ready to assist with any tasks. How can I help you today?",
        "I'll address that in a moment.",
        "I'm really glad to help.",
    ],
)
def test_words_that_merely_start_like_actions_are_not_claims(reply):
    """2026-10-30: "I'm ready" matched "read" and a greeting got a forced question."""
    assert not claims_action(reply)


# ---------------------------------------------------------------------------
#  2026-10-30, quick-ask: "No its ok" got the question it answered asked again;
#  "any meetings or tasks?" was answered "none" without looking; a misspelled
#  city sent the forecast to the current location.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "message,expected",
    [
        ("No its ok", True),
        ("no thanks", True),
        ("never mind", True),
        ("thank you gideon", True),
        ("Yes please", False),
        ("ok open notepad", False),
        ("No, set it for 5pm instead", False),
    ],
)
def test_declines(message, expected):
    assert is_decline(message) is expected


def test_a_claim_after_a_decline_is_cut_not_acted_on():
    reply_text = (
        "Right away sir. I've checked your schedule and task list, no meetings or pending items for today. "
        "Let me know if you'd like to add anything."
    )
    chat = _chat(reply_text)
    reply, trace = _run(Agent(get_db(), None, chat=chat), "No its ok")
    assert len(chat.calls) == 1, "no forced decision after a decline"
    assert reply == "Let me know if you'd like to add anything."


@pytest.mark.parametrize(
    "reply,expected",
    [
        ("Yes boss. I don’t see any scheduled meetings or pending tasks for today.", True),
        ("You have no reminders.", True),
        ("There is no RAM limit on that model.", False),
        ("No problem at all!", False),
    ],
)
def test_absence_claimed_without_looking(reply, expected):
    assert claims_action(reply) is expected


@pytest.mark.parametrize(
    "value,said,expected",
    [
        ("Barcelona", "what does the weahter look like in barcelone", True),
        ("Madrid", "weather in madird", True),
        ("London", "weather in linden", False),
        ("Mumbai", "weather in dubai", False),
    ],
)
def test_a_typo_is_still_the_users_word(value, said, expected):
    assert is_grounded(value, said) is expected



@pytest.mark.parametrize(
    "reply",
    [
        "Here are the top three reasons to learn Python.",
        "Your battery will last longer if you dim the screen.",
        "Here's how flash attention works.",
    ],
)
def test_explanations_are_not_readings(reply):
    assert not claims_action(reply)


def test_a_gerund_after_a_command_is_a_claim():
    """Eval 2026-10-30: "Opening dumbpipe in a new VS Code window." — nothing opened."""
    assert claims_action("Yes boss, opening dumbpipe in a new VS Code window.", command=True)
    assert not claims_action("Opening hours for the museum are 9 to 5.", command=False)
