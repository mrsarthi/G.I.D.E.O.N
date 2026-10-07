"""
G.I.D.E.O.N Intent Heuristics

Deterministic decisions a 3B model gets wrong on its own: whether this message
wants a tool at all, and whether "play X" means the play button or a song.
"""

from __future__ import annotations

import re
from typing import Any

from gideon.core.config import TOOL_GATING
from gideon.core.logs import get_logger
from gideon.tools.registry import trigger_words

log = get_logger("gideon.brain.intent")

# ---------------------------------------------------------------------------
#  Tool gating
# ---------------------------------------------------------------------------
# Words that signal the user wants something *done* rather than discussed.
# Each tool declares its own (see @tool(triggers=...)), so a new tool brings
# its vocabulary with it. Deliberately generous: a false positive only costs
# prefill, while a false negative means a command silently does nothing.
TOOL_INTENT_WORDS: frozenset[str] = frozenset(trigger_words())

# Arithmetic asked in words or symbols: "45 times 892", "12 * 7", "30% of 90".
_ARITHMETIC = re.compile(
    r"\d+\s*(?:[+\-*/×÷^%]|times|plus|minus|divided by|multiplied by|squared)\s*\d*",
    re.IGNORECASE,
)

# Multi-word phrases that imply action without a single keyword.
_TOOL_INTENT_PHRASES = (
    "check my", "check the", "how much", "what's my", "whats my", "what is my",
    "turn up", "turn down", "turn on", "turn off", "set the", "pull up",
    "bring up", "shut down", "open up",
    # Music requests that name no keyword: "put on some tame impala".
    "put on", "throw on", "listen to",
    # Factual/measurement questions routed to Wolfram Alpha. Kept as phrases
    # because the bare words ("how", "where") are ordinary conversation.
    "how far", "how many", "how much", "how tall", "how old", "how deep",
    "how fast", "how big", "how long is", "how heavy",
    "what is the distance", "where am i", "where is",
    # Market/weather phrasings with no distinctive keyword.
    "doing today", "going to rain", "going to snow",
)


def should_offer_tools(user_prompt: str) -> bool:
    """Decide whether to expose the tool schemas for this message.

    Returning False both prevents spurious side effects and strips ~2000
    tokens of schema from the prompt.
    """
    if not TOOL_GATING:
        return True

    text = (user_prompt or "").lower()
    if not text.strip():
        return False

    words = set(re.findall(r"[a-z']+", text))
    if words & TOOL_INTENT_WORDS:
        return True
    if any(phrase in text for phrase in _TOOL_INTENT_PHRASES):
        return True
    return bool(_ARITHMETIC.search(text))


# ---------------------------------------------------------------------------
#  Media intent routing
# ---------------------------------------------------------------------------
# "play X" is ambiguous: pressing the play button, or playing a named track.
# llama3.2:3b resolves it wrong — asked to "play loser by tame impala" it
# called control_media(play_pause), which toggled the media key and played
# nothing. Descriptions alone do not fix this on a 3B model, so the choice is
# corrected deterministically from the user's own words.

_MEDIA_VERBS = r"(?:play|put on|puton|listen to|queue|start|blast|throw on)"

# Words that mean "the transport button", not "a piece of music".
_TRANSPORT_ONLY = {
    "play", "pause", "resume", "stop", "next", "previous", "prev", "back",
    "skip", "forward", "track", "song", "music", "it", "this", "that", "the",
    "a", "an", "my", "some", "again", "please", "now", "one", "tune",
}

# Noise to strip off the end of an extracted query.
_QUERY_TRAILERS = re.compile(
    r"\b(?:on|in|via|using|through)?\s*"
    r"(?:youtube music|yt music|youtube|ytmusic|spotify)\b.*$|"
    r"\b(?:for me|please|thanks|thank you)\b\.?\s*$",
    re.IGNORECASE,
)


def extract_media_query(user_prompt: str) -> str | None:
    """Pull a named track/artist out of a play request.

    Returns None when the user meant the transport button rather than a
    specific piece of music.
    """
    text = (user_prompt or "").strip().lower()
    if not text:
        return None

    match = re.search(_MEDIA_VERBS + r"\s+(.+)", text)
    if not match:
        return None

    query = match.group(1).strip(" .,!?")
    # Strip "on youtube music", "for me", "please", etc.
    for _ in range(2):
        query = _QUERY_TRAILERS.sub("", query).strip(" .,!?")

    if not query:
        return None

    # If nothing survives except transport vocabulary, it was a button press.
    meaningful = [w for w in re.findall(r"[a-z0-9']+", query) if w not in _TRANSPORT_ONLY]
    if not meaningful:
        return None

    return query


def correct_media_tool_choice(
    name: str, args: dict[str, Any], user_prompt: str
) -> tuple[str, dict[str, Any]]:
    """Repair media tool selection using the user's actual wording."""
    named_track = extract_media_query(user_prompt)

    if name == "control_media" and named_track:
        log.info("Rerouting control_media -> play_youtube_music (%r)", named_track)
        return "play_youtube_music", {"query": named_track}

    if name == "play_youtube_music":
        query = str(args.get("query", "")).strip()
        if not named_track and not query:
            # "pause"/"next" misrouted to the music search.
            log.info("Rerouting empty play_youtube_music -> control_media")
            return "control_media", {"action": "play_pause"}
        if named_track and not query:
            # Model picked the right tool but dropped the argument.
            log.info("Recovered missing play query from prompt: %r", named_track)
            return name, {"query": named_track}

    return name, args
