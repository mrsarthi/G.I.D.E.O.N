"""Interception of tool calls that the model types out as plain text.

llama3.2:3b frequently ignores the native tool-call channel and prints
`{"name": ..., "parameters": ...}` as its answer. Observed live: asking it to
describe its memory system produced
`{"name": "explain_memory", "parameters": {"verbose": true}}`, which was
streamed to the UI and read aloud.
"""

from __future__ import annotations

import pytest

from orchestrator import parse_pseudo_tool_call


def test_intercepts_the_observed_failure():
    call = parse_pseudo_tool_call(
        '{"name": "explain_memory", "parameters": {"verbose": true, "length": 3}}'
    )
    assert call == {
        "name": "explain_memory",
        "arguments": {"verbose": True, "length": 3},
    }


def test_accepts_arguments_key():
    call = parse_pseudo_tool_call('{"name": "search_web", "arguments": {"query": "x"}}')
    assert call["name"] == "search_web"
    assert call["arguments"] == {"query": "x"}


def test_accepts_fenced_json():
    call = parse_pseudo_tool_call('```json\n{"name": "list_reminders"}\n```')
    assert call["name"] == "list_reminders"
    assert call["arguments"] == {}


def test_missing_parameters_defaults_to_empty():
    assert parse_pseudo_tool_call('{"name": "get_system_status"}')["arguments"] == {}


def test_non_dict_parameters_are_discarded():
    call = parse_pseudo_tool_call('{"name": "x", "parameters": "not an object"}')
    assert call["arguments"] == {}


def test_incomplete_json_returns_none_so_caller_keeps_buffering():
    assert parse_pseudo_tool_call('{"name": "search_web", "param') is None


@pytest.mark.parametrize(
    "text",
    [
        "Your GPU is at twelve percent, boss.",
        "",
        "   ",
        '{"not_a_name": true}',
        '{"name": 42}',
        "[1, 2, 3]",
        '{"name": ""}',
    ],
)
def test_ordinary_text_is_not_mistaken_for_a_tool_call(text):
    assert parse_pseudo_tool_call(text) is None


def test_json_object_without_a_name_is_left_as_speech():
    """A code sample in a reply must still be spoken, not swallowed."""
    assert parse_pseudo_tool_call('{"cpu": 12, "gpu": 40}') is None


def test_unknown_tool_name_still_intercepted_then_handled_by_dispatcher():
    """Interception must happen even for tools that do not exist, so the
    dispatcher can tell the model instead of the user hearing JSON."""
    from tools.registry import dispatch_tool_call

    call = parse_pseudo_tool_call('{"name": "explain_memory", "parameters": {}}')
    assert call is not None
    result = dispatch_tool_call(call["name"], call["arguments"])
    assert "no tool called" in result.lower()
