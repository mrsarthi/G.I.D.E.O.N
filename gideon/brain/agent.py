"""
G.I.D.E.O.N Agent

The one conversational loop every interface shares. The web API, Telegram,
the quick-ask window and the CLI all call Agent.stream / Agent.respond; none
of them talk to Ollama, the prompt builder or the tool registry directly.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from datetime import datetime
from typing import Any

import ollama

from gideon.brain.intent import correct_media_tool_choice, should_offer_tools
from gideon.brain.prompt import build_context_window, format_messages_for_ollama
from gideon.brain.toolcalls import (
    PSEUDO_CALL_CHAR_LIMIT,
    extract_chunk_content,
    extract_tool_calls,
    is_tool_call_candidate,
    json_start_index,
    parse_pseudo_tool_call,
)
from gideon.core.config import (
    CHAT_KEEP_ALIVE,
    CHAT_MODEL,
    MAX_TOOL_TURNS,
    OLLAMA_OPTIONS,
    REQUIRE_CONFIRMATION,
)
from gideon.core.context import current_channel, current_session
from gideon.core.logs import get_logger
from gideon.storage.db import DatabaseManager
from gideon.storage.memory import VectorMemory
from gideon.tools.registry import dispatch_tool_call, get_tool, tools_for_channel

log = get_logger("gideon.brain.agent")

OFFLINE_REPLY = (
    "I've lost contact with my local inference engine, boss. "
    "Check that Ollama is running."
)
EXHAUSTED_REPLY = "I wasn't able to finish that one — the tool chain ran too long."


def execute_tool(name: str, args: dict[str, Any], channel: str = "web") -> str:
    """Run a tool behind the confirmation gate, converting failures to text."""
    spec = get_tool(name)
    if REQUIRE_CONFIRMATION and spec is not None and spec.confirm:
        log.info("Tool %s blocked pending user confirmation", name)
        return (
            f"The '{name}' action needs the user's explicit confirmation before it can run. "
            f"Ask the user to confirm they want this, then wait for their reply."
        )
    try:
        return dispatch_tool_call(name, args, channel=channel)
    except Exception as exc:
        log.exception("Tool %s raised", name)
        return f"The {name} tool failed: {exc}"


class Agent:
    def __init__(
        self,
        db: DatabaseManager,
        memory: VectorMemory | None,
        chat: Callable[..., Any] = ollama.chat,
    ) -> None:
        self.db = db
        self.memory = memory
        # Injectable so tests can drive the loop without a running Ollama.
        self._chat = chat

    # ------------------------------------------------------------------
    #  Persistence
    # ------------------------------------------------------------------
    def _remember(self, message_id: int, content: str, session_id: str, role: str) -> None:
        if self.memory is not None:
            self.memory.add_async(
                message_id, content, session_id, role, datetime.now().isoformat()
            )

    def record_user_message(self, session_id: str, text: str) -> int:
        """Persist the incoming message before generating.

        stream() builds its context from SQLite. If the prompt is not stored
        first, the model answers the previous turn instead.
        """
        message_id = self.db.insert_message(session_id, "user", text)
        self._remember(message_id, text, session_id, "user")
        return message_id

    # ------------------------------------------------------------------
    #  The loop
    # ------------------------------------------------------------------
    def stream(self, session_id: str, user_prompt: str, channel: str = "web") -> Iterator[str]:
        """Stream a reply, executing tools in between passes when the model asks.

        Yields plain text deltas. Raises nothing: transport failures are
        converted into a spoken apology so an SSE stream always terminates
        cleanly.
        """
        db = self.db
        context_messages = build_context_window(db, session_id)
        messages = format_messages_for_ollama(db, self.memory, context_messages, user_prompt)

        # Read by tools (set_reminder) to learn where this turn came from.
        current_channel.set(channel)
        current_session.set(session_id)

        offer_tools = should_offer_tools(user_prompt)
        # Schemas are filtered per channel: a remote channel never even sees
        # the tools it is not allowed to run.
        channel_tools = tools_for_channel(channel)
        log.debug("Tools %s for this turn", "offered" if offer_tools else "withheld")

        for turn in range(MAX_TOOL_TURNS):
            full_ans = ""
            collected_tool_calls: list[dict[str, Any]] = []
            try:
                stream = self._chat(
                    model=CHAT_MODEL,
                    messages=messages,
                    tools=channel_tools if offer_tools else [],
                    stream=True,
                    options=OLLAMA_OPTIONS,
                    keep_alive=CHAT_KEEP_ALIVE,
                )

                held = ""          # Buffered text from the first brace onwards.
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

                    # Prose streams straight through; only text from a '{' or
                    # a code fence onwards is withheld, and only until it can
                    # be classified. Buffering every reply would delay the
                    # first token by the whole buffer and defeat sentence TTS.
                    if held:
                        held += token
                    else:
                        split_at = json_start_index(token)
                        if split_at is None:
                            full_ans += token
                            yield token
                            continue
                        prefix, held = token[:split_at], token[split_at:]
                        if prefix:
                            full_ans += prefix
                            yield prefix

                    pseudo = parse_pseudo_tool_call(held)
                    if is_tool_call_candidate(pseudo):
                        log.info("Intercepted text-emitted tool call: %s", pseudo["name"])
                        collected_tool_calls.append(pseudo)
                        held, suppress = "", True
                    elif len(held) >= PSEUDO_CALL_CHAR_LIMIT:
                        # Too long to be a tool call — release it as speech.
                        full_ans += held
                        yield held
                        held = ""

                # Anything still withheld when the stream ends.
                if held and not suppress:
                    pseudo = parse_pseudo_tool_call(held)
                    if is_tool_call_candidate(pseudo):
                        log.info("Intercepted end-of-stream tool call: %s", pseudo["name"])
                        collected_tool_calls.append(pseudo)
                    else:
                        full_ans += held
                        yield held

            except Exception:
                log.exception("Ollama stream failed on turn %d", turn)
                yield OFFLINE_REPLY
                return

            # No tools requested: this was the final answer.
            if not collected_tool_calls:
                if full_ans.strip():
                    asst_id = db.insert_message(session_id, "assistant", full_ans)
                    self._remember(asst_id, full_ans, session_id, "assistant")
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

                tool_result = execute_tool(name, args, channel=channel)

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
        yield EXHAUSTED_REPLY

    def respond(self, session_id: str, user_prompt: str, channel: str = "web") -> str:
        """Run a whole turn and return the full reply text."""
        return "".join(self.stream(session_id, user_prompt, channel))
