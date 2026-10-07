"""
G.I.D.E.O.N Tool-Call Parsing

Reading Ollama's response shapes, and recognising tool calls a small model
typed out as prose instead of using the native tool-call channel.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypeGuard

from gideon.core.logs import get_logger
from gideon.tools.registry import VALID_TOOL_NAMES

log = get_logger("gideon.brain.toolcalls")

# How much of a JSON-looking reply to buffer before deciding it is really
# prose that happens to start with a brace.
PSEUDO_CALL_CHAR_LIMIT = 400


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


def _call_from_object(obj: Any) -> dict[str, Any] | None:
    if not isinstance(obj, dict):
        return None
    name = obj.get("name") or obj.get("function") or obj.get("tool")
    if isinstance(name, dict):
        name = name.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    args = obj.get("parameters") or obj.get("arguments") or obj.get("args") or {}
    return {"name": name.strip(), "arguments": args if isinstance(args, dict) else {}}


def _try_json(snippet: str) -> dict[str, Any] | None:
    try:
        return _call_from_object(json.loads(snippet))
    except (json.JSONDecodeError, TypeError):
        return None


def parse_pseudo_tool_call(text: str) -> dict[str, Any] | None:
    """Recognise a tool call that the model emitted as plain text anywhere in its output.

    Small models regularly ignore the native tool-call channel and print
    `{"name": "...", "parameters": {...}}` or introductory prose followed by
    JSON. Ollama hands that to us as content, so without this it gets streamed
    to the UI and read aloud by TTS or sent over Telegram — the exact failure
    RULE 4 of the system prompt tries to prevent.

    Returns None when the text does not contain a valid JSON tool call object.
    """
    if not text or not text.strip():
        return None

    candidate = text.strip()

    # 1. Direct JSON or fenced code block
    if candidate.startswith("```"):
        candidate = candidate.strip("`").strip()
        if candidate[:4].lower() == "json":
            candidate = candidate[4:].strip()

    if candidate.startswith("{"):
        found = _try_json(candidate)
        if found:
            return found

    # 2. Embedded markdown JSON block in text
    for match in re.finditer(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text, re.IGNORECASE):
        found = _try_json(match.group(1))
        if found:
            return found

    # 3. Embedded JSON object in conversational prose
    #    (e.g., "boss, here's a JSON for the function call:\n\n{...}")
    for match in re.finditer(r"(\{\s*\"(?:name|function|tool)\"[\s\S]*?\})", text):
        found = _try_json(match.group(1))
        if found:
            return found

    # 4. Fallback: try parsing from the first '{' to last '}'
    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace > first_brace:
        found = _try_json(text[first_brace : last_brace + 1])
        if found:
            return found

    return None


def json_start_index(text: str) -> int | None:
    """Find the index where a JSON object or code block starts in the text."""
    for idx, char in enumerate(text):
        if char == "{" or text[idx:].startswith("```"):
            return idx
    return None


def is_tool_call_candidate(pseudo: dict[str, Any] | None) -> TypeGuard[dict[str, Any]]:
    """Whether a parsed pseudo tool call names a real tool worth intercepting."""
    return pseudo is not None and pseudo.get("name") in VALID_TOOL_NAMES
