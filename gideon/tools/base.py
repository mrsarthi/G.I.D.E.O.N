"""
G.I.D.E.O.N Tool Declarations

A tool is declared once, next to its implementation:

    @tool(
        "get_weather",
        "Fetch live weather ...",
        Param("location", "City name. Omit for current location.",
              aliases=("city", "place"), default=None),
        triggers={"weather", "forecast", "rain"},
    )
    def get_weather(location: str | None = None) -> str: ...

That one declaration produces the JSON schema Ollama sees, the argument
normalisation the dispatcher applies, the channel and confirmation policy, and
the words that make tool gating offer it. Previously these lived in four
places (schema list, handler wrapper, handler table, config sets) that had
already started to drift.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

_MISSING_VALUES = (None, "")


@dataclass(frozen=True)
class Param:
    """One model-facing parameter.

    `aliases` exist because small models rename parameters freely ("cmd" for
    "command"). The first non-empty of name + aliases wins; otherwise
    `default`. `kwarg` maps the schema name onto a differently named Python
    argument.
    """

    name: str
    description: str
    type: str = "string"
    required: bool = False
    enum: tuple[str, ...] | None = None
    aliases: tuple[str, ...] = ()
    default: Any = ""
    kwarg: str | None = None

    def schema(self) -> dict[str, Any]:
        body: dict[str, Any] = {"type": self.type}
        if self.enum:
            body["enum"] = list(self.enum)
        body["description"] = self.description
        return body

    def resolve(self, arguments: dict[str, Any]) -> Any:
        for key in (self.name, *self.aliases):
            if key in arguments and arguments[key] not in _MISSING_VALUES:
                return self._coerce(arguments[key])
        return self.default

    def _coerce(self, value: Any) -> Any:
        if self.type != "integer":
            return value
        try:
            return int(value)
        except (TypeError, ValueError):
            return self.default


@dataclass
class ToolSpec:
    name: str
    description: str
    params: tuple[Param, ...]
    fn: Callable[..., str]
    # Hidden tools are dispatchable (CLI, programmatic use) but never offered
    # to the model.
    exposed: bool = True
    # Channels that may neither see nor run this tool. Telegram is remote and
    # guarded only by a user-ID allowlist, so it never gets code execution.
    blocked_channels: frozenset[str] = frozenset()
    # Waits for the user's approval when GIDEON_REQUIRE_CONFIRMATION is on.
    confirm: bool = False
    # Can change the machine. Needs approval whenever outside content (web
    # pages, documents, clipboard) entered the turn first, whatever the
    # confirmation setting — that content may carry injected instructions.
    privileged: bool = False
    # Returns text from outside the user's control. Marked as data for the
    # model, and it taints the rest of the turn (see `privileged`).
    untrusted_output: bool = False
    # Words that signal the user wants this tool; feeds brain/intent.py.
    triggers: frozenset[str] = frozenset()
    # Pass the requesting channel through as `channel=` (for auditing).
    wants_channel: bool = False

    def schema(self) -> dict[str, Any]:
        parameters: dict[str, Any] = {
            "type": "object",
            "properties": {p.name: p.schema() for p in self.params},
        }
        required = [p.name for p in self.params if p.required]
        if required:
            parameters["required"] = required
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": parameters,
            },
        }

    def bind(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """Normalise model-supplied arguments into Python keyword arguments."""
        return {p.kwarg or p.name: p.resolve(arguments) for p in self.params}

    def available_on(self, channel: str) -> bool:
        return channel not in self.blocked_channels

    def run(self, arguments: dict[str, Any], channel: str = "web") -> str:
        kwargs = self.bind(arguments)
        if self.wants_channel:
            kwargs["channel"] = channel
        return self.fn(**kwargs)


# Populated as tool modules are imported; tools/registry.py imports them all.
_REGISTRY: dict[str, ToolSpec] = {}


def tool(
    name: str,
    description: str,
    *params: Param,
    exposed: bool = True,
    blocked_channels: set[str] | frozenset[str] = frozenset(),
    confirm: bool = False,
    privileged: bool = False,
    untrusted_output: bool = False,
    triggers: set[str] | frozenset[str] = frozenset(),
    wants_channel: bool = False,
) -> Callable[[Callable[..., str]], Callable[..., str]]:
    """Register the decorated function as a tool. Returns it unchanged."""

    def register(fn: Callable[..., str]) -> Callable[..., str]:
        if name in _REGISTRY:
            raise ValueError(f"Tool {name!r} is declared twice")
        _REGISTRY[name] = ToolSpec(
            name=name,
            description=description,
            params=params,
            fn=fn,
            exposed=exposed,
            blocked_channels=frozenset(blocked_channels),
            confirm=confirm,
            privileged=privileged,
            untrusted_output=untrusted_output,
            triggers=frozenset(triggers),
            wants_channel=wants_channel,
        )
        return fn

    return register
