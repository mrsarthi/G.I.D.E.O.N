"""Prompt assembly guarantees.

The prefix cache and the tool-leak fix both live in message ordering, so
these assertions are what stop the regressions from coming back.
"""

from __future__ import annotations

from gideon.brain.prompt import (
    STATIC_SYSTEM_PROMPT,
    estimate_tokens,
    format_messages_for_ollama,
    is_context_worthy,
)


class StubDB:
    """Minimal stand-in for DatabaseManager."""

    def get_user_profile(self):
        return {"user_name": "boss", "ongoing_projects": "G.I.D.E.O.N"}


def build(history):
    return format_messages_for_ollama(StubDB(), None, history, "what's my GPU at?")


def test_static_prompt_is_first_and_unmodified():
    """Index 0 must be byte-identical every turn or prefill is re-done."""
    messages = build([{"role": "user", "content": "hello"}])
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == STATIC_SYSTEM_PROMPT


def test_static_prompt_contains_nothing_volatile():
    lowered = STATIC_SYSTEM_PROMPT.lower()
    for volatile in ("current local time", "2026", "user name:"):
        assert volatile not in lowered


def test_live_context_rides_on_the_newest_user_turn():
    """Not a system message (templates hoist those into the cached prefix) and
    not a separate user message (qwen2.5 then stops calling tools)."""
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "second"},
        {"role": "user", "content": "third"},
    ]
    messages = build(history)
    live = [i for i, m in enumerate(messages) if "[LIVE CONTEXT]" in m["content"]]
    assert live == [len(messages) - 1]
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"].startswith("third\n\n[LIVE CONTEXT]")
    # Only one system message, and it is the static one.
    assert [m["role"] for m in messages].count("system") == 1


def test_history_is_replayed_unchanged():
    history = [
        {"role": "user", "content": "older"},
        {"role": "assistant", "content": "reply"},
        {"role": "user", "content": "newest"},
    ]
    messages = build(history)
    assert messages[1:3] == history[:2]


def test_prefix_grows_monotonically_across_turns():
    """Turn N's prefix must remain a prefix of turn N+1's."""
    turn_one = build([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}])
    turn_two = build(
        [
            {"role": "user", "content": "a"},
            {"role": "assistant", "content": "b"},
            {"role": "user", "content": "c"},
        ]
    )
    stable_one = [m for m in turn_one if "[LIVE CONTEXT]" not in m["content"]]
    for i, msg in enumerate(stable_one):
        assert turn_two[i] == msg


def test_empty_history_still_includes_user_prompt():
    messages = build([])
    assert messages[-1]["role"] == "user"
    assert messages[-1]["content"].startswith("what's my GPU at?\n\n[LIVE CONTEXT]")


def test_tool_bookkeeping_never_replays_as_speech():
    """Assistant rows saying '[Tool Call: ...]' taught the model to narrate."""
    assert not is_context_worthy("assistant", "[Tool Call: get_system_status]")
    assert not is_context_worthy("tool_call", "[Tool Call: launch_app]")
    assert not is_context_worthy("tool", "CPU Utilization: 12%")
    assert is_context_worthy("assistant", "Your GPU is at 12%, boss.")


def test_blank_messages_are_dropped():
    assert not is_context_worthy("user", "   ")


def test_token_estimate_is_monotonic():
    assert estimate_tokens("") == 0
    assert estimate_tokens("a" * 400) > estimate_tokens("a" * 40)
