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
    # Notes and tasks. "to-do" tokenises as "to" + "do", so the tool's own
    # "todo" trigger never sees it.
    "to-do", "to do list", "write down", "jot down", "make a note",
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


# ---------------------------------------------------------------------------
#  Claimed actions
# ---------------------------------------------------------------------------
# Measured in F-12: qwen2.5 answered 31 tool requests with "I will check your
# GPU temperature right away" and never called a tool; qwen3:4b-instruct did
# it 10 times ("I'll set that reminder for you"). To the user that is a false
# success. A reply matching these patterns, with no tool run in the turn, is
# sent back to the model once (brain/agent.py).
_ACTION_VERBS = (
    r"check|set|add|open|launch|play|search|look|copy|remind|save|note|mute|"
    r"unmute|turn|run|read|find|pull|get|fetch|start|create|put|mark|list|show|"
    r"schedule|queue|skip|pause|resume|lower|raise|make|made|write|written|jot|"
    r"retriev|grab|bring"
)
# Things only a tool could know. "Here are your reminders: 1. Meeting at 3pm…"
# without a tool call is invented data (measured on qwen3:4b-instruct, F-15).
_PERSONAL_DATA = (
    r"reminders?|to-?do|tasks?|notes?|clipboard|volume|battery|ram|cpu|gpu|"
    r"files?|documents?|ip(?:\s+address)?|system\s+status"
)
_CLAIM_PATTERNS = (
    # "I'll check", "I will set", "I've added", "Let me open", "I'm going to play"
    re.compile(
        r"\b(?:i['’]?ll|i\s+will|i['’]?m\s+going\s+to|let\s+me|i['’]?ve|i\s+have|i\s+am|i['’]?m)"
        r"\s+(?:now\s+|go\s+ahead\s+and\s+|just\s+|quickly\s+|also\s+)?"
        rf"(?:{_ACTION_VERBS})\w*\b",
        re.IGNORECASE,
    ),
    # "I can check your GPU temperature right now." — an offer made instead of
    # the call (eval, 2026-10-12). Needs "right now/away" so describing what
    # G.I.D.E.O.N can do ("I can set reminders for you") is not a claim.
    re.compile(
        rf"\bi\s+can\s+(?:{_ACTION_VERBS})\w*\b[^.!?\n]{{0,60}}\bright\s+(?:now|away)\b",
        re.IGNORECASE,
    ),
    # "Checking your GPU now…", "Yes boss. Opening Notepad."
    re.compile(
        r"^\W*(?:(?:yes|right\s+away|sure|certainly|okay|ok|of\s+course|on\s+it)\b[^.!?\n]{0,15}[.!,]?\s*)?"
        r"(?:checking|setting|opening|launching|playing|searching|adding|saving|copying|"
        r"muting|running|reading|fetching|pulling|scheduling)\b",
        re.IGNORECASE,
    ),
    # "Here are your current reminders: …", "Here's what's on your to-do list"
    re.compile(
        r"\bhere(?:['’]s|\s+is|\s+are)\s+(?:what['’]?s\s+on\s+)?(?:your|the)\s+"
        r"(?:current\s+|upcoming\s+|latest\s+|saved\s+)?"
        rf"(?:{_PERSONAL_DATA})\b",
        re.IGNORECASE,
    ),
    # Reporting the result instead of promising it: "Notepad is now open",
    # "Your reminder has been set", "Calculator launched successfully"
    # (measured on qwen3:4b-instruct: "Yes boss. Notepad is now open." with
    # no tool call).
    re.compile(
        r"\b(?:is|are)\s+now\s+(?:open(?:ed)?|launched|running|playing|paused|muted|unmuted|"
        r"set|saved|added|copied|scheduled|on|off|at\s+\d+)\b"
        r"|\b(?:has|have)\s+(?:now\s+|just\s+)?been\s+(?:successfully\s+)?"
        r"(?:opened|launched|started|set|saved|added|copied|scheduled|created|muted|unmuted|"
        r"paused|resumed|played|noted|completed|marked|run|adjusted|changed)\b"
        r"|\b(?:opened|launched|started|saved|added|copied|scheduled|set)\s+successfully\b"
        r"|\bsuccessfully\s+(?:opened|launched|started|saved|added|copied|scheduled|set)\b",
        re.IGNORECASE,
    ),
    # Live facts only a tool could know. The live context carries the time
    # but no weather or prices, so "The current weather in Varanasi is mostly
    # cloudy with a high of 28°C" with no tool call is invented (seen live,
    # 100% on that phrasing).
    re.compile(
        r"\bweather\s+(?:in|for|at|around)\s+[^.?!\n]{1,40}?\s+(?:is|looks|will\s+be)\b"
        r"|\bchance\s+of\s+(?:light\s+|heavy\s+)?(?:rain|showers|drizzle|snow|thunderstorms?)\b"
        r"|\b(?:high|low)\s+of\s+-?\d+"
        r"|\bcurrently\s+trading\s+at\b"
        # The opposite failure: refusing what a tool can do. "I don't have
        # access to real-time weather data" while get_weather is offered.
        r"|\b(?:don['’]?t|do\s+not|can['’]?t|cannot)\s+(?:have\s+)?(?:access\s+to\s+|provide\s+|check\s+)?"
        r"(?:the\s+)?(?:real[-\s]?time|live|current)\s+(?:weather|temperature|data|information|prices?)\b",
        re.IGNORECASE,
    ),
)


def claims_action(reply: str) -> bool:
    """Whether a reply promises or reports doing something a tool would do."""
    text = (reply or "").strip()
    return bool(text) and any(p.search(text) for p in _CLAIM_PATTERNS)
