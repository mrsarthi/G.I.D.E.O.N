"""
G.I.D.E.O.N Orchestration Engine

Design notes that are easy to break:

* Prefix cache. Ollama only reuses cached prefill while the *leading* tokens
  are byte-identical between turns. Anything volatile (the clock, recalled
  memories, the profile) therefore lives at the END of the message list, not
  at index 0. Putting the clock in the system prompt re-prefills the whole
  context every minute.
* Token budget. Ollama prepends the tool schemas to the prompt, so the
  history budget must be num_ctx minus schemas, minus recalled memories,
  minus room for the answer. See config.PROMPT_TOKEN_BUDGET.
* Tool bookkeeping never re-enters the context as assistant speech, or the
  model learns to say "[Tool Call: ...]" out loud.
"""

from __future__ import annotations

import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Iterator

import ollama

from config import (
    CHARS_PER_TOKEN,
    CHAT_KEEP_ALIVE,
    CHAT_MODEL,
    CONFIRM_REQUIRED_TOOLS,
    HISTORY_MESSAGE_LIMIT,
    MAX_TOOL_TURNS,
    MEMORY_RESERVE,
    NUM_CTX,
    OLLAMA_OPTIONS,
    PROMPT_TOKEN_BUDGET,
    RESPONSE_RESERVE,
    TOOL_SCHEMA_RESERVE,
    REQUIRE_CONFIRMATION,
    TOOL_GATING,
    get_logger,
)
from db import DatabaseManager, get_db
from memory import addMemory, initMemory, queryMemory
from tools.registry import OLLAMA_TOOLS, dispatch_tool_call

log = get_logger("gideon.orchestrator")

# Roles that are replayed as conversation. Tool bookkeeping is excluded.
_CONTEXT_ROLES = {"user", "assistant"}
# Rows written by older builds that stored tool calls as assistant speech.
_LEGACY_TOOL_PREFIXES = ("[Tool Call:", "[Tool Result]")

# How much of a JSON-looking reply to buffer before deciding it is really
# prose that happens to start with a brace.
PSEUDO_CALL_CHAR_LIMIT = 400

_memory_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chroma_writer")


def _measure_tool_schema_tokens() -> int:
    """Cost of the tool schemas Ollama prepends to every tool-enabled prompt.

    Measured rather than guessed: config.TOOL_SCHEMA_RESERVE was set for 11
    tools (~1,220 tokens). At 18 tools the schemas cost ~2,090, and the stale
    constant let the assembled prompt exceed num_ctx — at which point Ollama
    truncates from the front and silently discards the system prompt.
    """
    return len(json.dumps(OLLAMA_TOOLS)) // CHARS_PER_TOKEN


TOOL_SCHEMA_TOKENS = _measure_tool_schema_tokens()

# Budget derived from the real schema size, so adding a tool cannot overflow
# the context window without the budget shrinking to match.
EFFECTIVE_PROMPT_BUDGET = max(
    1024, NUM_CTX - TOOL_SCHEMA_TOKENS - RESPONSE_RESERVE - MEMORY_RESERVE
)

if TOOL_SCHEMA_TOKENS > TOOL_SCHEMA_RESERVE:
    log.warning(
        "Tool schemas cost %d tokens but TOOL_SCHEMA_RESERVE is %d; "
        "using measured value (history budget %d tokens).",
        TOOL_SCHEMA_TOKENS,
        TOOL_SCHEMA_RESERVE,
        EFFECTIVE_PROMPT_BUDGET,
    )


def addMemoryAsync(collection, messageId, content, sessionId, role, timestamp):
    """Submit a ChromaDB write to a background thread (non-blocking)."""
    _memory_executor.submit(
        addMemory, collection, messageId, content, sessionId, role, timestamp
    )


def estimateTokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


# ---------------------------------------------------------------------------
#  Ollama response shape helpers
# ---------------------------------------------------------------------------
def extract_tool_calls(response: Any) -> list[dict[str, Any]]:
    """Extract tool calls from an Ollama ChatResponse object or dict."""
    msg = getattr(response, "message", None)
    if msg is None and isinstance(response, dict):
        msg = response.get("message")
    if not msg:
        return []

    raw_calls = getattr(msg, "tool_calls", None)
    if raw_calls is None and isinstance(msg, dict):
        raw_calls = msg.get("tool_calls")
    if not raw_calls:
        return []

    parsed_calls = []
    for call in raw_calls:
        fn = getattr(call, "function", None) or (
            call.get("function") if isinstance(call, dict) else None
        )
        if not fn:
            continue
        name = getattr(fn, "name", None) or (
            fn.get("name") if isinstance(fn, dict) else ""
        )
        args = getattr(fn, "arguments", None) or (
            fn.get("arguments") if isinstance(fn, dict) else {}
        )
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                log.warning("Tool %s sent unparseable arguments: %s", name, args)
                args = {}
        parsed_calls.append({"name": name, "arguments": args})

    return parsed_calls


def parse_pseudo_tool_call(text: str) -> dict[str, Any] | None:
    """Recognise a tool call that the model emitted as plain text.

    Small models regularly ignore the native tool-call channel and just print
    `{"name": "...", "parameters": {...}}` into their answer. Ollama hands
    that to us as content, so without this it gets streamed to the UI and
    read aloud by the TTS — the exact failure RULE 4 tries to prevent.

    Returns None when the text is not (yet) a complete JSON object.
    """
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.strip("`").strip()
        if candidate[:4].lower() == "json":
            candidate = candidate[4:].strip()
    if not candidate.startswith("{"):
        return None

    try:
        obj = json.loads(candidate)
    except json.JSONDecodeError:
        return None  # Possibly incomplete; caller keeps buffering.

    if not isinstance(obj, dict):
        return None

    name = obj.get("name") or obj.get("function") or obj.get("tool")
    if not isinstance(name, str) or not name:
        return None

    args = obj.get("parameters") or obj.get("arguments") or {}
    return {"name": name, "arguments": args if isinstance(args, dict) else {}}


def extract_chunk_content(chunk: Any) -> str:
    """Extract the text delta from an Ollama stream chunk."""
    msg = getattr(chunk, "message", None)
    if msg is None and isinstance(chunk, dict):
        msg = chunk.get("message")
    if not msg:
        return ""
    content = getattr(msg, "content", None)
    if content is None and isinstance(msg, dict):
        content = msg.get("content", "")
    return content or ""


# ---------------------------------------------------------------------------
#  Prompt assembly
# ---------------------------------------------------------------------------
# Index 0 and byte-stable for the life of the process. Nothing dynamic here,
# ever — this is the block the prefix cache is built on.
STATIC_SYSTEM_PROMPT = """You are G.I.D.E.O.N (Guided Intelligence Does Everything One Needs), an elite personal AI assistant inspired by JARVIS.

[OPERATIONAL DIRECTIVES & TOOL GUARDRAILS]
- You are concise, precise, direct, and conversational—an ambient AI assistant like JARVIS. Never break character.
- RULE 1 (GENERAL CHAT, EXPLANATIONS & FOLLOW-UPS): When the user asks general technical/conceptual questions, requests summaries, or says things like "in brief please", "summarize that", or "hi", DO NOT invoke any tools. Respond directly using your intelligence and conversation history.
- RULE 2 (EXPLICIT USER COMMANDS ONLY): Only invoke a tool when the user gives a clear operational command (e.g., checking PC hardware diagnostics, reminders, opening an app, checking/copying to the Windows clipboard, searching the web, adjusting PC audio volume, controlling media, searching or playing tracks on YouTube Music, or reading a local document/PDF).
- RULE 3 (INVISIBLE WEB SEARCHING): When the user instructs you to search the internet or check online info, ALWAYS use search_web. NEVER attempt to open a browser window for the user. Read the background text results and explain the findings directly in natural speech.
- RULE 4 (NEVER LEAK TOOL JSON IN SPEECH): Never say things like "I would recommend a function call to" and never output raw JSON tool schemas in your spoken responses. Always reply directly in natural human language.
- RULE 5 (DATE/TIME AWARENESS): The current local date and time is supplied to you in the live context block. Never invoke a tool to check the time or date.
- RULE 6 (PLAYING MUSIC BY NAME): When the user asks to play a specific song, track, album, or artist by name (e.g., "play Loser by Tame Impala"), ALWAYS use play_youtube_music. NEVER use control_media for named music queries.
"""


def build_live_context_block(db: DatabaseManager, memories: str) -> str:
    """Volatile context. Goes at the END of the message list, never the start."""
    profile = db.get_user_profile()
    user_name = profile.get("user_name", "boss")
    projects = profile.get("ongoing_projects", "G.I.D.E.O.N AI Development")
    now = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

    block = f"""[LIVE CONTEXT]
- User Name: {user_name}
- Preferred Addressing Style: Always address the user naturally as "{user_name}", "boss", or "sir".
- Ongoing Projects: {projects}
- Current Local Time: {now}"""

    if memories:
        block += f"\n\n[RELEVANT MEMORIES FROM PAST CONVERSATIONS]\n{memories}"

    return block


def _is_context_worthy(role: str, content: str) -> bool:
    """Filter tool bookkeeping out of replayed conversation history."""
    if role not in _CONTEXT_ROLES:
        return False
    if content.startswith(_LEGACY_TOOL_PREFIXES):
        return False
    return bool(content.strip())


def build_context_window(db: DatabaseManager, session_id: str) -> list[dict[str, Any]]:
    """Assemble recent conversation, newest-first, within the token budget."""
    rows = db.get_sliding_window(session_id, limit=HISTORY_MESSAGE_LIMIT)

    budget = EFFECTIVE_PROMPT_BUDGET - estimateTokens(STATIC_SYSTEM_PROMPT)
    context_messages: list[dict[str, Any]] = []
    token_count = 0

    for role, content in reversed(rows):
        if not _is_context_worthy(role, content):
            continue
        msg_tokens = estimateTokens(content)
        if token_count + msg_tokens > budget:
            break
        context_messages.insert(0, {"role": role, "content": content})
        token_count += msg_tokens

    log.debug(
        "Context: %d messages, ~%d tokens (budget %d)",
        len(context_messages),
        token_count,
        budget,
    )
    return context_messages


def auto_retrieve_memories(collection: Any, user_prompt: str) -> str:
    """Recall relevant past messages, trimmed to the memory token reserve."""
    if not user_prompt.strip():
        return ""
    try:
        results = queryMemory(collection, user_prompt)
    except Exception as exc:
        log.error("Memory retrieval failed: %s", exc)
        return ""

    if not results:
        return ""

    snippets: list[str] = []
    used = 0
    for m in results:
        content = m.get("content", "").strip()
        if not content:
            continue
        cost = estimateTokens(content)
        if used + cost > MEMORY_RESERVE:
            break
        snippets.append(content)
        used += cost

    return "\n---\n".join(snippets)


def format_messages_for_ollama(
    db: DatabaseManager,
    collection: Any,
    context_messages: list[dict[str, Any]],
    user_prompt: str,
) -> list[dict[str, Any]]:
    """Build the final message list, cache-stable prefix first, volatile last.

    Layout: [static system] [older history] [live context] [newest user turn]
    Everything up to and including the older history is byte-identical to the
    previous turn, so Ollama reuses its prefill.
    """
    memories = auto_retrieve_memories(collection, user_prompt)
    live_block = build_live_context_block(db, memories)

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": STATIC_SYSTEM_PROMPT}
    ]

    if context_messages:
        messages.extend(context_messages[:-1])
        messages.append({"role": "system", "content": live_block})
        messages.append(context_messages[-1])
    else:
        messages.append({"role": "system", "content": live_block})
        messages.append({"role": "user", "content": user_prompt})

    return messages


# ---------------------------------------------------------------------------
#  Tool gating
# ---------------------------------------------------------------------------
# Words that signal the user wants something *done* rather than discussed.
# Deliberately generous: a false positive only costs prefill, while a false
# negative means a command silently does nothing.
_TOOL_INTENT_WORDS = {
    # apps / system
    "open", "launch", "run", "start", "close", "app", "application",
    "cpu", "gpu", "ram", "vram", "disk", "drive", "battery", "temperature",
    "temp", "hardware", "diagnostics", "diagnostic", "status", "telemetry",
    "system", "performance", "usage", "specs",
    # search & info
    "search", "google", "look", "lookup", "browse", "internet", "web",
    "online", "news", "weather", "forecast", "latest", "current", "stock",
    "stocks", "crypto", "market", "shares", "ticker", "finance", "financial",
    "hackernews", "hn", "ip", "geolocation", "isp", "wolfram", "calculate",
    "convert", "equation",
    # clipboard
    "clipboard", "copy", "paste", "copied",
    # audio / media
    "volume", "mute", "unmute", "louder", "quieter", "sound", "audio",
    "play", "pause", "resume", "skip", "track", "song", "music", "spotify",
    "next", "previous", "listen", "youtube", "yt", "album", "playlist",
    "artist", "tune", "queue", "blast",
    # reminders
    "remind", "reminder", "reminders", "alarm", "schedule", "alert",
    # files
    "file", "files", "folder", "directory", "find", "locate", "read",
    "document", "pdf", "open up",
    # weather (Open-Meteo)
    "forecast", "rain", "raining", "snow", "sunny", "cloudy", "humidity",
    "hot", "cold", "windy", "degrees", "celsius", "fahrenheit", "climate",
    # markets (yfinance)
    "stock", "stocks", "share", "shares", "ticker", "quote", "market",
    "markets", "nasdaq", "dow", "crypto", "bitcoin", "btc", "eth",
    "ethereum", "price", "trading", "portfolio", "earnings", "dividend",
    # news / HackerNews
    "headlines", "hackernews", "hn", "story", "stories", "article",
    # geolocation (IP-API)
    "ip", "location", "where", "geolocation", "isp", "timezone",
    # computation / Wolfram Alpha
    "calculate", "compute", "convert", "solve", "equation", "distance",
    "population", "wolfram",
}

# Arithmetic asked in words or symbols: "45 times 892", "12 * 7", "30% of 90".
_ARITHMETIC = re.compile(
    r"\d+\s*(?:[+\-*/×÷^%]|times|plus|minus|divided by|multiplied by|squared)\s*\d*",
    re.IGNORECASE,
)

# Multi-word phrases that imply action without a single keyword.
_TOOL_INTENT_PHRASES = (
    "check my", "check the", "how much", "what's my", "whats my", "what is my",
    "turn up", "turn down", "turn on", "turn off", "set the", "pull up",
    "bring up", "shut down",
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

    Returning False both prevents spurious side effects and strips ~1200
    tokens of schema from the prompt.
    """
    if not TOOL_GATING:
        return True

    text = (user_prompt or "").lower()
    if not text.strip():
        return False

    words = set(re.findall(r"[a-z']+", text))
    if words & _TOOL_INTENT_WORDS:
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
#  Tool execution
# ---------------------------------------------------------------------------
def execute_tool(name: str, args: dict[str, Any], collection: Any = None) -> str:
    """Run a tool behind the confirmation gate, converting failures to text."""
    if REQUIRE_CONFIRMATION and name in CONFIRM_REQUIRED_TOOLS:
        log.info("Tool %s blocked pending user confirmation", name)
        return (
            f"The '{name}' action needs the user's explicit confirmation before it can run. "
            f"Ask the user to confirm they want this, then wait for their reply."
        )
    try:
        return dispatch_tool_call(name, args, chroma_collection=collection)
    except Exception as exc:
        log.exception("Tool %s raised", name)
        return f"The {name} tool failed: {exc}"


# ---------------------------------------------------------------------------
#  Agent loop
# ---------------------------------------------------------------------------
def run_agent_stream(
    db: DatabaseManager, collection: Any, session_id: str, user_prompt: str
) -> Iterator[str]:
    """Stream a reply, executing tools in between passes when the model asks.

    Yields plain text deltas. Raises nothing: transport failures are converted
    into a spoken apology so the SSE stream always terminates cleanly.
    """
    context_messages = build_context_window(db, session_id)
    messages = format_messages_for_ollama(db, collection, context_messages, user_prompt)

    offer_tools = should_offer_tools(user_prompt)
    log.debug("Tools %s for this turn", "offered" if offer_tools else "withheld")

    for turn in range(MAX_TOOL_TURNS):
        try:
            stream = ollama.chat(
                model=CHAT_MODEL,
                messages=messages,
                tools=OLLAMA_TOOLS if offer_tools else [],
                stream=True,
                options=OLLAMA_OPTIONS,
                keep_alive=CHAT_KEEP_ALIVE,
            )

            full_ans = ""
            collected_tool_calls: list[dict[str, Any]] = []
            held = ""          # Buffered opening tokens that may be JSON.
            resolved = False   # True once we know this turn is real speech.
            suppress = False   # True after intercepting a text tool call.

            for chunk in stream:
                calls = extract_tool_calls(chunk)
                if calls:
                    collected_tool_calls.extend(calls)
                    continue

                token = extract_chunk_content(chunk)
                if not token or suppress:
                    continue
                # Some GGUF chat templates emit a stray role header first.
                if not full_ans and not held and token.strip().lower() == "assistant":
                    continue

                # Withhold a reply that opens like JSON until we know whether
                # it is speech or a tool call the model typed out by hand.
                if not resolved and not full_ans:
                    if (held + token).lstrip().startswith(("{", "```")):
                        held += token
                        pseudo = parse_pseudo_tool_call(held)
                        if pseudo:
                            log.info("Intercepted text-emitted tool call: %s", pseudo["name"])
                            collected_tool_calls.append(pseudo)
                            held, suppress, resolved = "", True, True
                            continue
                        if len(held) < PSEUDO_CALL_CHAR_LIMIT:
                            continue
                        # Too long to be a tool call — release it as speech.
                        token, held = held, ""
                    elif held:
                        token, held = held + token, ""
                    resolved = True

                full_ans += token
                yield token

            # Stream ended mid-buffer: emit whatever was withheld.
            if held and not suppress:
                full_ans += held
                yield held

        except Exception as exc:
            log.exception("Ollama stream failed on turn %d", turn)
            yield (
                "I've lost contact with my local inference engine, boss. "
                "Check that Ollama is running."
            )
            return

        # No tools requested: this was the final answer.
        if not collected_tool_calls:
            if full_ans.strip():
                asst_id = db.insert_message(session_id, "assistant", full_ans)
                addMemoryAsync(
                    collection,
                    asst_id,
                    full_ans,
                    session_id,
                    "assistant",
                    datetime.now().isoformat(),
                )
            return

        # Tools requested: execute, feed results back, and loop.
        for tool_call in collected_tool_calls:
            name = tool_call.get("name", "")
            args = tool_call.get("arguments", {})
            if not isinstance(args, dict):
                args = {}

            # A 3B model cannot reliably tell "press play" from "play <song>".
            name, args = correct_media_tool_choice(name, args, user_prompt)
            log.info("Tool call: %s(%s)", name, args)

            tool_result = execute_tool(name, args, collection)

            # Stored as 'tool_call'/'tool' so build_context_window never
            # replays them as something G.I.D.E.O.N said out loud.
            db.insert_message(
                session_id,
                "tool_call",
                f"[Tool Call: {name}]",
                tool_name=name,
                tool_args=json.dumps(args),
            )
            db.insert_message(session_id, "tool", tool_result, tool_name=name)

            messages.append(
                {
                    "role": "assistant",
                    "content": full_ans,
                    "tool_calls": [{"function": {"name": name, "arguments": args}}],
                }
            )
            messages.append({"role": "tool", "content": str(tool_result), "name": name})

    log.warning("Tool loop exhausted %d turns without a final answer", MAX_TOOL_TURNS)
    yield "I wasn't able to finish that one, boss — the tool chain ran too long."


def run_chat_loop(
    db: DatabaseManager, collection: Any, session_id: str, user_prompt: str
) -> str:
    """Synchronously run the agent loop for a turn and return the full text response."""
    return "".join(run_agent_stream(db, collection, session_id, user_prompt))


def run():
    """Terminal CLI interactive chat mode."""
    # The Windows console defaults to cp1252, which cannot encode the box
    # drawing characters below or most model output (em dashes, degree signs).
    for stream_obj in (sys.stdout, sys.stderr):
        try:
            stream_obj.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    db = get_db()
    collection = initMemory()

    session_id = db.create_session("Terminal Session")
    print("\n\033[36m╔══════════════════════════════════════╗")
    print("║   G.I.D.E.O.N  Agentic Engine v2.0   ║")
    print("╚══════════════════════════════════════╝\033[0m\n")

    while True:
        try:
            inp = input("\033[36mYou: \033[0m").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye, boss.")
            break

        if not inp:
            continue
        if inp.lower() in ("/quit", "exit"):
            break

        user_id = db.insert_message(session_id, "user", inp)
        addMemoryAsync(collection, user_id, inp, session_id, "user", datetime.now().isoformat())

        print("\033[32mG.I.D.E.O.N:\033[0m ", end="", flush=True)
        for token in run_agent_stream(db, collection, session_id, inp):
            print(token, end="", flush=True)
        print()

    db.close()


if __name__ == "__main__":
    run()
