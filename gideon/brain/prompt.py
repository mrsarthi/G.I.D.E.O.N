"""
G.I.D.E.O.N Prompt Assembly

Design notes that are easy to break:

* Prefix cache. Ollama only reuses cached prefill while the *leading* tokens
  are byte-identical between turns. Anything volatile (the clock, recalled
  memories, the profile) therefore lives at the END of the message list, not
  at index 0. Putting the clock in the system prompt re-prefills the whole
  context every minute.
* Token budget. Ollama prepends the tool schemas to the prompt, so the
  history budget must be num_ctx minus schemas, minus recalled memories,
  minus room for the answer.
* Tool bookkeeping never re-enters the context as assistant speech, or the
  model learns to say "[Tool Call: ...]" out loud.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from gideon.core.config import (
    CHARS_PER_TOKEN,
    HISTORY_MESSAGE_LIMIT,
    MEMORY_RESERVE,
    NUM_CTX,
    RESPONSE_RESERVE,
    TOOL_SCHEMA_RESERVE,
)
from gideon.core.logs import get_logger
from gideon.storage.db import DatabaseManager
from gideon.storage.memory import TOOL_BOOKKEEPING_PREFIXES, VectorMemory
from gideon.tools.registry import OLLAMA_TOOLS

log = get_logger("gideon.brain.prompt")

# Roles that are replayed as conversation. Tool bookkeeping is excluded.
_CONTEXT_ROLES = {"user", "assistant"}


def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def _measure_tool_schema_tokens() -> int:
    """Cost of the tool schemas Ollama prepends to every tool-enabled prompt.

    Measured rather than guessed: TOOL_SCHEMA_RESERVE was once set for 11
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


# Index 0 and byte-stable for the life of the process. Nothing dynamic here,
# ever — this is the block the prefix cache is built on.
STATIC_SYSTEM_PROMPT = """You are G.I.D.E.O.N (Guided Intelligence Does Everything One Needs), an elite personal AI assistant inspired by JARVIS.

[OPERATIONAL DIRECTIVES & TOOL GUARDRAILS]
- You are concise, precise, direct, and conversational—an ambient AI assistant like JARVIS. Never break character.
- RULE 1 (GENERAL CHAT, EXPLANATIONS & FOLLOW-UPS): When the user asks general technical/conceptual questions, requests summaries, or says things like "in brief please", "summarize that", or "hi", DO NOT invoke any tools. Respond directly using your intelligence and conversation history.
- RULE 2 (EXPLICIT USER COMMANDS ONLY): Only invoke a tool when the user gives a clear operational command (e.g., checking PC hardware diagnostics, reminders, opening an app, checking/copying to the Windows clipboard, searching the web, adjusting PC audio volume, controlling media, searching or playing tracks on YouTube Music, or reading a local document/PDF).
- RULE 3 (INVISIBLE WEB SEARCHING): When the user instructs you to search the internet or check online info, ALWAYS use search_web. NEVER attempt to open a browser window for the user. Read the background text results and explain the findings directly in natural speech.
- RULE 4 (NEVER LEAK TOOL JSON IN SPEECH): Never say things like "I would recommend a function call to" or "here's a JSON for the function call", and NEVER output raw JSON tool schemas or objects in your spoken responses. Always reply directly in natural human language.
- RULE 5 (DATE/TIME AWARENESS): The current local date and time is supplied to you in the live context block. Never invoke a tool to check the time or date.
- RULE 6 (PLAYING MUSIC BY NAME): When the user asks to play a specific song, track, album, or artist by name (e.g., "play Loser by Tame Impala"), ALWAYS use play_youtube_music. NEVER use control_media for named music queries.
- RULE 7 (ADDRESSING STYLE & FORBIDDEN ENDINGS): Randomly alternate addressing the user as "boss" or "sir" across turns. Address the user AT MOST ONCE per response, strictly near the beginning (e.g., "Yes boss", "Right away sir", "Certainly boss"). NEVER put a comma immediately after "boss" or "sir" (e.g., write "Yes boss" instead of "Yes, boss," so TTS speech flows naturally without a pause). NEVER put "boss" or "sir" at the end of a sentence or at the end of your response.
"""


def build_live_context_block(db: DatabaseManager, memories: str) -> str:
    """Volatile context. Goes at the END of the message list, never the start."""
    profile = db.get_user_profile()
    user_name = profile.get("user_name", "boss")
    projects = profile.get("ongoing_projects", "G.I.D.E.O.N AI Development")
    now = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

    block = f"""[LIVE CONTEXT]
- User Name: {user_name}
- Address Formats: Randomly alternate between "boss" and "sir" in your response openings.
- Addressing Rules:
  1. Address the user AT MOST ONCE per turn, strictly near the beginning.
  2. NEVER append "boss" or "sir" at the end of a sentence or end of response.
  3. DO NOT place a comma after "boss" or "sir" (write "Yes boss..." or "Certainly sir..." without a comma after the title).
- Ongoing Projects: {projects}
- Current Local Time: {now}"""

    if memories:
        block += f"\n\n[RELEVANT MEMORIES FROM PAST CONVERSATIONS]\n{memories}"

    return block


def is_context_worthy(role: str, content: str) -> bool:
    """Filter tool bookkeeping out of replayed conversation history."""
    if role not in _CONTEXT_ROLES:
        return False
    # Rows written by older builds that stored tool calls as assistant speech.
    if content.startswith(TOOL_BOOKKEEPING_PREFIXES):
        return False
    return bool(content.strip())


def build_context_window(db: DatabaseManager, session_id: str) -> list[dict[str, Any]]:
    """Assemble recent conversation, newest-first, within the token budget."""
    rows = db.get_sliding_window(session_id, limit=HISTORY_MESSAGE_LIMIT)

    budget = EFFECTIVE_PROMPT_BUDGET - estimate_tokens(STATIC_SYSTEM_PROMPT)
    context_messages: list[dict[str, Any]] = []
    token_count = 0

    for row in reversed(rows):
        if not is_context_worthy(row.role, row.content):
            continue
        msg_tokens = estimate_tokens(row.content)
        if token_count + msg_tokens > budget:
            break
        context_messages.insert(0, {"role": row.role, "content": row.content})
        token_count += msg_tokens

    log.debug(
        "Context: %d messages, ~%d tokens (budget %d)",
        len(context_messages),
        token_count,
        budget,
    )
    return context_messages


def retrieve_memories(memory: VectorMemory | None, user_prompt: str) -> str:
    """Recall relevant past messages, trimmed to the memory token reserve."""
    if memory is None or not user_prompt.strip():
        return ""
    try:
        results = memory.query(user_prompt)
    except Exception as exc:
        log.error("Memory retrieval failed: %s", exc)
        return ""

    snippets: list[str] = []
    used = 0
    for m in results:
        content = m.get("content", "").strip()
        if not content:
            continue
        cost = estimate_tokens(content)
        if used + cost > MEMORY_RESERVE:
            break
        snippets.append(content)
        used += cost

    return "\n---\n".join(snippets)


def format_messages_for_ollama(
    db: DatabaseManager,
    memory: VectorMemory | None,
    context_messages: list[dict[str, Any]],
    user_prompt: str,
) -> list[dict[str, Any]]:
    """Build the final message list, cache-stable prefix first, volatile last.

    Layout: [static system] [older history] [live context] [newest user turn]
    Everything up to and including the older history is byte-identical to the
    previous turn, so Ollama reuses its prefill.
    """
    memories = retrieve_memories(memory, user_prompt)
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
