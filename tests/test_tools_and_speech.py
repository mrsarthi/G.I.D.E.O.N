"""Tool dispatch safety, memory filtering, and TTS sentence chunking."""

from __future__ import annotations

import pytest

from memory import isWorthRemembering
from tools.desktop import KNOWN_APPS, launch_app
from tools.registry import OLLAMA_TOOLS, VALID_TOOL_NAMES, dispatch_tool_call


# ---------------------------------------------------------------------------
#  Tool dispatch
# ---------------------------------------------------------------------------
def test_every_schema_has_a_handler():
    """A schema without a handler is a tool the model can call into a wall."""
    from tools.registry import _HANDLERS

    assert VALID_TOOL_NAMES <= set(_HANDLERS)


def test_schemas_are_well_formed():
    for tool in OLLAMA_TOOLS:
        fn = tool["function"]
        assert fn["name"] and fn["description"]
        assert fn["parameters"]["type"] == "object"


def test_unknown_tool_returns_guidance_not_an_exception():
    result = dispatch_tool_call("teleport", {})
    assert "no tool called" in result.lower()


def test_dispatch_survives_non_dict_arguments():
    result = dispatch_tool_call("get_time_date", "not a dict")
    assert isinstance(result, str) and result


def test_argument_aliases_are_accepted():
    """Small models rename parameters; the dispatcher normalises them."""
    from tools.registry import _first_arg

    assert _first_arg({"query": "x"}, "file_name_or_extension", "query") == "x"
    assert _first_arg({}, "missing", default="fallback") == "fallback"
    assert _first_arg({"text": ""}, "text", default="fallback") == "fallback"


# ---------------------------------------------------------------------------
#  Launch allowlist
# ---------------------------------------------------------------------------
def test_unlisted_app_is_refused():
    result = launch_app("definitely-not-a-real-program")
    assert "not on the approved application list" in result


@pytest.mark.parametrize(
    "payload",
    [
        "notepad & calc",
        "notepad && del /f /s /q C:\\*",
        "notepad; rm -rf /",
        "notepad | powershell -c evil",
    ],
)
def test_shell_metacharacters_cannot_reach_a_shell(payload):
    """The old implementation passed this straight to `start` with shell=True."""
    assert payload.lower() not in KNOWN_APPS
    result = launch_app(payload)
    assert "not on the approved application list" in result


def test_known_app_is_recognised():
    result = launch_app("notepad")
    assert "not on the approved application list" not in result


# ---------------------------------------------------------------------------
#  Memory filtering
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("junk", ["ok", "thanks", "hi", "yep", "  cool  ", "lol"])
def test_filler_is_not_indexed(junk):
    assert not isWorthRemembering(junk, "user")


def test_tool_output_is_not_indexed():
    assert not isWorthRemembering("CPU Utilization: 12%", "tool")
    assert not isWorthRemembering("[Tool Call: get_system_status]", "assistant")


def test_substantive_message_is_indexed():
    assert isWorthRemembering(
        "I'm building a voice-first assistant on a GTX 1650 Ti with 4GB of VRAM.",
        "user",
    )


# ---------------------------------------------------------------------------
#  Streaming speech chunking
# ---------------------------------------------------------------------------
def test_incomplete_sentence_is_held_back():
    from server import split_ready_sentences

    sentences, remainder = split_ready_sentences("Your GPU is running")
    assert sentences == []
    assert remainder == "Your GPU is running"


def test_complete_sentence_is_released_early():
    """This is what lets speech start before generation finishes."""
    from server import split_ready_sentences

    sentences, remainder = split_ready_sentences(
        "Your GPU is at twelve percent, boss. Memory looks"
    )
    assert sentences == ["Your GPU is at twelve percent, boss."]
    assert remainder == "Memory looks"


def test_multiple_sentences_split_in_order():
    from server import split_ready_sentences

    sentences, _ = split_ready_sentences(
        "First observation here. Second observation here. Third one trails"
    )
    assert sentences == ["First observation here.", "Second observation here."]


def test_short_fragments_merge_forward():
    """A stray 'Yes.' should not become its own audio file."""
    from server import split_ready_sentences

    sentences, _ = split_ready_sentences("Yes. That is correct about the GPU. Next")
    assert all(len(s) >= 12 for s in sentences)
    assert sentences[0].startswith("Yes.")


def test_question_and_exclamation_terminate_sentences():
    from server import split_ready_sentences

    sentences, _ = split_ready_sentences("Shall I proceed with that? Awaiting orders")
    assert sentences == ["Shall I proceed with that?"]


def test_search_uses_the_maintained_ddgs_package():
    """duckduckgo_search is abandoned: it imports fine but returns zero
    results for most queries, with no exception to signal the failure."""
    from tools.web_and_clipboard import DDGS

    assert DDGS is not None
    assert DDGS.__module__.startswith("ddgs"), (
        "falling back to the deprecated duckduckgo_search package"
    )
