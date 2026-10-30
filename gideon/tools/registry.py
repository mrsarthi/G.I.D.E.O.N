"""
G.I.D.E.O.N Tool Registry

Collects every @tool declaration (tools/base.py) and dispatches model calls to
them. Importing this module imports each tool module, which is what registers
them; add a new module to _TOOL_MODULES and nothing else.

Every exposed schema is prepended to each tool-enabled prompt by Ollama, so it
costs context on every turn — see brain/prompt.py. Keep descriptions tight.
"""

from __future__ import annotations

import importlib
from typing import Any

from gideon.core.config import POLICY_FILE
from gideon.core.logs import get_logger
from gideon.tools.base import _REGISTRY, ToolSpec

log = get_logger("gideon.tools.registry")

_TOOL_MODULES = (
    "desktop",
    "reminders",
    "notes",
    "web_and_clipboard",
    "media_and_docs",
    "yt_music",
    "weather",
    "finance",
    "football",
    "news",
    "geolocation",
    "knowledge",
    "terminal",
    "samay",
)

for _module in _TOOL_MODULES:
    importlib.import_module(f"gideon.tools.{_module}")

# policy.toml is the source of truth for what each tool may do; the values in
# the @tool declarations are fallbacks (tools/policy.py).
from gideon.tools.policy import apply_policy, load_policy  # noqa: E402

apply_policy(_REGISTRY, load_policy(POLICY_FILE))

# The order schemas are offered in. Small models are order-biased — the
# specific play_youtube_music must precede the generic control_media or "play
# <song>" presses the play button — and a stable order keeps Ollama's prefix
# cache valid across restarts. Tools not listed here are offered last.
_SCHEMA_ORDER = (
    "get_system_status",
    "set_reminder",
    "list_reminders",
    "save_note",
    "list_notes",
    "complete_task",
    "launch_app",
    "search_files",
    "search_web",
    "get_clipboard",
    "set_clipboard",
    "control_volume",
    "play_youtube_music",
    "control_media",
    "read_document",
    "search_youtube_music",
    "get_weather",
    "get_financial_data",
    "get_news",
    "get_football",
    "get_hackernews",
    "get_ip_info",
    "query_wolfram_alpha",
    "run_terminal_command",
)


def get_tool(name: str) -> ToolSpec | None:
    return _REGISTRY.get((name or "").strip().lower())


def all_tools() -> list[ToolSpec]:
    return list(_REGISTRY.values())


def exposed_tools() -> list[ToolSpec]:
    rank = {name: i for i, name in enumerate(_SCHEMA_ORDER)}
    exposed = [spec for spec in _REGISTRY.values() if spec.exposed]
    return sorted(exposed, key=lambda spec: rank.get(spec.name, len(rank)))


# Schemas offered to the model. Derived from the declarations so the two can
# never drift apart.
OLLAMA_TOOLS: list[dict[str, Any]] = [spec.schema() for spec in exposed_tools()]
VALID_TOOL_NAMES: set[str] = {spec.name for spec in exposed_tools()}


def tools_for_channel(channel: str) -> list[dict[str, Any]]:
    """Schemas this channel is permitted to see.

    Withholding the schema is stronger than refusing the call: the model
    cannot pick a tool it was never shown.
    """
    return [spec.schema() for spec in exposed_tools() if spec.available_on(channel)]


def trigger_words() -> set[str]:
    """Every word any tool declared as signalling intent to use it."""
    words: set[str] = set()
    for spec in _REGISTRY.values():
        words |= spec.triggers
    return words


def dispatch_tool_call(tool_name: str, arguments: Any, channel: str = "web") -> str:
    """Route a tool call to its implementation and return a string result."""
    name = (tool_name or "").strip().lower()
    spec = _REGISTRY.get(name)

    # Second line of defence: even if a schema leaked into the prompt, a
    # blocked channel cannot execute the tool.
    if spec is not None and not spec.available_on(channel):
        log.warning("Tool %r blocked for channel %r", name, channel)
        return (
            f"The '{name}' tool is not available over {channel}. "
            f"Ask me from the desktop interface instead."
        )

    if spec is None:
        log.warning("Model requested unknown tool %r", tool_name)
        return (
            f"There is no tool called '{tool_name}'. "
            f"Available tools: {', '.join(sorted(VALID_TOOL_NAMES))}."
        )

    if not isinstance(arguments, dict):
        arguments = {}

    return spec.run(arguments, channel=channel)
