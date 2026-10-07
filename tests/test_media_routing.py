"""Media intent routing.

Observed live: "could you please play loser by tame impala" made llama3.2:3b
call control_media(play_pause). The media key toggled, nothing played, and
G.I.D.E.O.N claimed the track was playing. Tool descriptions alone do not fix
this on a 3B model, so the choice is corrected from the user's own wording.
"""

from __future__ import annotations

import pytest

from gideon.brain.intent import correct_media_tool_choice, extract_media_query, should_offer_tools


# ---------------------------------------------------------------------------
#  Query extraction
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("could you please play loser by tame impala", "loser by tame impala"),
        ("play loser by tame impala", "loser by tame impala"),
        ("play Starboy", "starboy"),
        ("put on the strokes", "the strokes"),
        ("listen to bohemian rhapsody", "bohemian rhapsody"),
        ("play redbone on youtube music", "redbone"),
        ("play let it happen for me", "let it happen"),
        ("play borderline please", "borderline"),
        ("queue up dracula by tame impala", "up dracula by tame impala"),
    ],
)
def test_named_tracks_are_extracted(prompt, expected):
    assert extract_media_query(prompt) == expected


@pytest.mark.parametrize(
    "prompt",
    [
        "pause",
        "pause the music",
        "resume playback",
        "next track",
        "skip this song",
        "play",
        "play it again",
        "play the next track",
        "stop the music",
        "go back a track",
        "",
        "   ",
    ],
)
def test_transport_commands_yield_no_query(prompt):
    assert extract_media_query(prompt) is None


def test_non_media_message_yields_no_query():
    assert extract_media_query("what is my GPU running at") is None


# ---------------------------------------------------------------------------
#  Tool choice correction
# ---------------------------------------------------------------------------
def test_the_exact_reported_failure_is_repaired():
    """control_media(play_pause) on a named track becomes a real play."""
    name, args = correct_media_tool_choice(
        "control_media",
        {"action": "play_pause"},
        "could you please play loser by tame impala",
    )
    assert name == "play_youtube_music"
    assert args == {"query": "loser by tame impala"}


def test_genuine_transport_command_is_left_alone():
    name, args = correct_media_tool_choice(
        "control_media", {"action": "play_pause"}, "pause the music"
    )
    assert name == "control_media"
    assert args == {"action": "play_pause"}


def test_next_track_is_left_alone():
    name, args = correct_media_tool_choice(
        "control_media", {"action": "next"}, "skip to the next track"
    )
    assert (name, args) == ("control_media", {"action": "next"})


def test_correct_tool_with_good_args_is_untouched():
    name, args = correct_media_tool_choice(
        "play_youtube_music", {"query": "loser tame impala"}, "play loser by tame impala"
    )
    assert name == "play_youtube_music"
    assert args == {"query": "loser tame impala"}


def test_missing_query_is_recovered_from_the_prompt():
    """Model picked the right tool but dropped the argument."""
    name, args = correct_media_tool_choice(
        "play_youtube_music", {}, "play redbone by childish gambino"
    )
    assert name == "play_youtube_music"
    assert args == {"query": "redbone by childish gambino"}


def test_music_search_without_any_track_falls_back_to_transport():
    name, args = correct_media_tool_choice("play_youtube_music", {}, "pause")
    assert name == "control_media"
    assert args == {"action": "play_pause"}


def test_unrelated_tools_are_never_rewritten():
    name, args = correct_media_tool_choice(
        "get_system_status", {}, "play loser by tame impala"
    )
    assert (name, args) == ("get_system_status", {})


# ---------------------------------------------------------------------------
#  Gating must let music requests through
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "prompt",
    [
        "could you please play loser by tame impala",
        "put on some tame impala",
        "listen to redbone",
        "play my liked songs playlist",
        "search youtube music for tame impala albums",
    ],
)
def test_music_requests_reach_the_tools(prompt):
    assert should_offer_tools(prompt) is True


# ---------------------------------------------------------------------------
#  Schema ordering
# ---------------------------------------------------------------------------
def test_specific_play_tool_precedes_the_generic_one():
    """Small models are order-biased; the specific tool must come first."""
    from gideon.tools.registry import OLLAMA_TOOLS

    names = [t["function"]["name"] for t in OLLAMA_TOOLS]
    assert names.index("play_youtube_music") < names.index("control_media")


# ---------------------------------------------------------------------------
#  Malformed action strings from the model
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "action,expected_fragment",
    [
        ("pause_pause", "playback"),        # observed live from llama3.2:3b
        ("media_next_track", "next track"),
        ("play-pause", "playback"),
        ("PLAY_PAUSE", "playback"),
        ("prev_track", "previous"),
        ("media_stop", "Stopped"),
    ],
)
def test_malformed_media_actions_are_normalised(action, expected_fragment, monkeypatch):
    """A rejected action left the user's 'pause' command doing nothing."""

    import gideon.tools.media_and_docs as md

    sent = []
    monkeypatch.setattr(
        md.ctypes, "windll",
        type("W", (), {"user32": type("U", (), {
            "keybd_event": staticmethod(lambda vk, a, b, c: sent.append(vk))
        })()})(),
        raising=False,
    )
    result = md.control_media(action)
    assert "Unknown media action" not in result
    assert expected_fragment.lower() in result.lower()
    assert sent, "no media key was sent"


def test_genuinely_unknown_action_is_still_rejected():
    import gideon.tools.media_and_docs as md

    assert "Unknown media action" in md.control_media("teleport")
