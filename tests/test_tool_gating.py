"""Tool gating.

Observed live with llama3.2:3b: "Say thanks in one short sentence" triggered
set_clipboard, silently overwriting the user's clipboard, and "Say goodbye"
triggered get_system_status. Each spurious call costs a wasted round trip.
"""

from __future__ import annotations

import pytest

from orchestrator import should_offer_tools


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


def test_gating_can_be_disabled():
    """With gating off the behaviour reverts to always offering tools."""
    import orchestrator

    original = orchestrator.TOOL_GATING
    try:
        orchestrator.TOOL_GATING = False
        assert should_offer_tools("hi") is True
    finally:
        orchestrator.TOOL_GATING = original
