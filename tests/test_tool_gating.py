"""Tool gating.

Observed live with llama3.2:3b: "Say thanks in one short sentence" triggered
set_clipboard, silently overwriting the user's clipboard, and "Say goodbye"
triggered get_system_status. Each spurious call costs a wasted round trip.
"""

from __future__ import annotations

import pytest

from gideon.brain.intent import should_offer_tools


@pytest.mark.parametrize(
    "message",
    [
        # The exact prompts that misfired in testing.
        "Say thanks in one short sentence.",
        "Say goodbye in one short sentence.",
        "Say hello in one short sentence.",
        "Say yes in one short sentence.",
        # Ordinary conversation and explanation requests.
        "hi",
        "in brief please",
        "summarize that for me",
        "who are you?",
        "explain how transformers work",
        "that was helpful, appreciate it",
        "tell me a joke",
    ],
)
def test_conversational_messages_withhold_tools(message):
    assert should_offer_tools(message) is False


@pytest.mark.parametrize(
    "message",
    [
        "open notepad",
        "launch chrome for me",
        "what is my GPU running at right now?",
        "check the CPU temperature",
        "how much RAM am I using",
        "search the web for piper tts voices",
        "google the latest ollama release",
        "what's the weather today",
        "copy that to my clipboard",
        "read my clipboard",
        "set the volume to 40",
        "mute the sound",
        "pause the music",
        "skip to the next track",
        "remind me to call mum tomorrow at 9am",
        "list my reminders",
        "find main.py on my disk",
        "read that pdf on my desktop",
        "turn down the volume",
        "check my battery",
    ],
)
def test_operational_messages_offer_tools(message):
    assert should_offer_tools(message) is True


def test_empty_message_withholds_tools():
    assert should_offer_tools("") is False
    assert should_offer_tools("   ") is False


def test_matching_is_case_insensitive():
    assert should_offer_tools("OPEN NOTEPAD") is True


def test_keyword_must_be_a_whole_word():
    """'copy' should trigger; 'copyright' in prose should not."""
    assert should_offer_tools("what does copyright law say about that") is False


def test_gating_can_be_disabled(monkeypatch):
    """With gating off the behaviour reverts to always offering tools."""
    from gideon.brain import intent

    monkeypatch.setattr(intent, "TOOL_GATING", False)
    assert should_offer_tools("hi") is True


def test_every_tool_contributes_its_trigger_words():
    """Gating vocabulary comes from the tool declarations, so a new tool that
    declares no triggers can never be offered and would silently do nothing."""
    from gideon.brain.intent import TOOL_INTENT_WORDS
    from gideon.tools.registry import exposed_tools

    for spec in exposed_tools():
        assert spec.triggers, f"{spec.name} declares no trigger words"
        assert spec.triggers <= TOOL_INTENT_WORDS


# ---------------------------------------------------------------------------
#  The five tools added later must be reachable through the gate
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "message",
    [
        # weather (Open-Meteo)
        "what's the weather in Tokyo?",
        "is it going to rain tomorrow",
        "how cold is it outside",
        # markets (yfinance)
        "how is Bitcoin doing today?",
        "AAPL quote",
        "check stock price for TSLA",
        "whats the market doing",
        # HackerNews
        "show me top stories on hacker news",
        "latest tech headlines",
        # geolocation (IP-API)
        "what is my IP and location?",
        "where am i right now",
        # Wolfram Alpha
        "what is 45 times 892",
        "12 * 7",
        "how far is the moon",
        "convert 30 celsius to fahrenheit",
        "ask wolfram alpha the distance to mars",
    ],
)
def test_new_tool_requests_reach_the_tools(message):
    """Gating was written for 11 tools; five more were added afterwards."""
    assert should_offer_tools(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "how does attention work",
        "explain how a transformer is trained",
        "what do you think about that",
        "who are you",
    ],
)
def test_broad_question_words_still_do_not_open_tools(message):
    """'how' and 'where' are ordinary conversation, not tool intent."""
    assert should_offer_tools(message) is False
