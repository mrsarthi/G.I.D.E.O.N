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
    r"retriev|grab|bring|trigger|switch|send|sent|load|skip|stop|resum|adjust|chang|clos|launch|press|hit"
)
# Words that start like an action verb but aren't one: "I'm ready" matched
# "read" (2026-10-30, a greeting got a forced follow-up question), and
# "I'll address that" / "I'll listen" would match "add" / "list".
_NOT_ACTIONS = (
    r"ready|really|realiz|realis|address|addition|listen|settl|openly|playful|"
    r"showcas|noteworth|lookout|market|startl|copyright"
)
# An action verb in any inflection: "check", "checking", "checked".
_ACTION_WORD = rf"(?!{_NOT_ACTIONS})(?:{_ACTION_VERBS})\w*\b"
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
        r"\s+(?:now\s+|go\s+ahead\s+and\s+|just\s+|quickly\s+|also\s+)?" + _ACTION_WORD,
        re.IGNORECASE,
    ),
    # "I can check your GPU temperature right now." — an offer made instead of
    # the call (eval, 2026-10-12). Needs "right now/away" so describing what
    # G.I.D.E.O.N can do ("I can set reminders for you") is not a claim.
    re.compile(
        rf"\bi\s+can\s+{_ACTION_WORD}[^.!?\n]{{0,60}}\bright\s+(?:now|away)\b",
        re.IGNORECASE,
    ),
    # "Checking your GPU now…", "Yes boss. Opening Notepad." The gerund needs
    # an object — "Running in the mornings is a great habit" is advice
    # (eval, 2026-10-27) — and a name only counts capitalised.
    re.compile(
        r"^\W*(?:(?:yes|right\s+away|sure|certainly|okay|ok|of\s+course|on\s+it)\b[^.!?\n]{0,15}[.!,]?\s*)?"
        r"(?:checking|setting|opening|launching|playing|searching|adding|saving|copying|"
        r"muting|running|reading|fetching|pulling|scheduling)\s+"
        r"(?:the|your|a|an|it|that|this|some|now|up|for\s+(?:the|your|a|an|you)|(?-i:[A-Z][\w.+-]*))\b",
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
    # A bare past tense opening a sentence: "Paused the music, sir." (seen
    # 2026-10-27 with no tool call), "Opened Notepad.". Not "Set": it reads the
    # same as an instruction ("2. **Set a consistent time**" in running tips);
    # after a command, the state words below still catch "set".
    # The object is required — "Set theory is…" and "Added sugar…" are not
    # claims — and is matched case-sensitively so a name counts ("Notepad").
    re.compile(
        r"(?:^|[.!?]\s+|\n)\W*(?:(?:[Yy]es|[Dd]one|[Oo]kay|[Oo]k|[Ss]ure|[Aa]lright|[Rr]ight)\b[^.!?\n]{0,15}"
        r"[.!,—-]?\s*)?"
        r"(?:[Oo]pened|[Ll]aunched|[Ss]tarted|[Pp]aused|[Rr]esumed|[Pp]layed|[Mm]uted|[Uu]nmuted|[Ss]topped|"
        r"[Ss]kipped|[Aa]dded|[Ss]aved|[Cc]opied|[Ss]cheduled|[Tt]urned|[Rr]aised|[Ll]owered|"
        r"[Cc]losed|[Cc]reated|[Nn]oted|[Mm]arked|[Cc]ompleted)\s+"
        r"(?:the|your|a|an|it|that|this|[A-Z][\w.+-]*)\b"
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


_SENTENCES = re.compile(r"[^.!?\n]+(?:[.!?]+|\n|$)")

# After a command, a statement reporting any of these states is a claim:
# "Calculator is open, sir.", "Music paused, sir." (2026-10-27: six of six
# such replies, no tool called, none caught by the patterns above). In
# conversation the same words are innocent ("the shop is open until 9"),
# so they only count when the user's message was a command.
_STATE_WORDS = re.compile(
    r"\b(?:open(?:ed)?|launch(?:ed)?|running|playing|paused|muted|unmuted|stopped|resumed|"
    r"closed|started|done|all\s+set|set|saved|added|copied|scheduled|skipped|"
    r"turned\s+(?:up|down|on|off))\b",
    re.IGNORECASE,
)


# A request to act, not a question that mentions an action: "open
# calculator", "can you pause the music", "hey gideon, set a reminder…" — but
# not "I want to start running in the mornings, any tips?" (2026-10-27: tools
# were offered for its trigger words, and the advice it got was taken for a
# claim and replaced by a reminder question).
_COMMAND = re.compile(
    r"^\W*(?:(?:hey|ok|okay|yo)\s+gideon\W*)?(?:please\s+|kindly\s+|just\s+)?"
    r"(?:(?:can|could|would|will)\s+you\s+(?:please\s+)?|i\s+(?:want|need)\s+you\s+to\s+|go\s+ahead\s+and\s+)?"
    rf"(?!{_NOT_ACTIONS})(?:{_ACTION_VERBS}|review)\w*\b",
    re.IGNORECASE,
)


def is_command(user_prompt: str) -> bool:
    """Whether the message asks G.I.D.E.O.N to do something, imperatively."""
    return bool(_COMMAND.match(user_prompt or ""))


def claims_action(reply: str, command: bool = False) -> bool:
    """Whether a reply promises or reports doing something a tool would do.

    `command`: the user's message asked for an action (tools were offered).
    Then any statement — not a question — reporting a state counts too.
    """
    text = (reply or "").strip()
    if not text:
        return False
    if any(p.search(text) for p in _CLAIM_PATTERNS):
        return True
    return command and any(
        _STATE_WORDS.search(s) and not s.rstrip().endswith("?") for s in _SENTENCES.findall(text)
    )


# Said instead when a reply was nothing but a false claim.
NOT_DONE_REPLY = "I haven't done that yet — it didn't go through."
# What is left after cutting a claim, if it says nothing ("Yes boss.").
_FILLER_ONLY = re.compile(
    r"^\W*(?:(?:yes|sure|certainly|okay|ok|alright|right|done|of\s+course|boss|sir|"
    r"ready\s+for\s+your\s+work)\W*)*$",
    re.IGNORECASE,
)


def strip_claims(reply: str, command: bool = False) -> str:
    """The reply without its sentences claiming what no tool did.

    Used when a claim was held back and nothing could replace it — a forced
    decision was already spent, or a tool refused. The rest (usually a
    question) is kept; a reply that was only a claim becomes NOT_DONE_REPLY.
    """
    sentences = [s.strip() for s in _SENTENCES.findall(reply or "") if s.strip()]
    # Filler goes with the claim it introduced ("Yes boss. <claim>").
    kept = [s for s in sentences if not claims_action(s, command) and not _FILLER_ONLY.match(s)]
    return " ".join(kept) or NOT_DONE_REPLY
