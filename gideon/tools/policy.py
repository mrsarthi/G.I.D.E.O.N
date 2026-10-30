"""
G.I.D.E.O.N Capability Policy

Loads `policy.toml` (PRD §5.6, §9.2) and applies it to the registered tools.
The file is the single, auditable source of each tool's capability class,
confirmation rule and channel restrictions; the values in each `@tool`
declaration are only fallbacks for a tool the file does not mention (which
the test suite treats as an error).

What stays in code: `untrusted_output` — whether a tool *returns* text from
outside the user's control is a fact about its implementation, not a policy
choice.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from gideon.core.logs import get_logger
from gideon.tools.base import ToolSpec

log = get_logger("gideon.tools.policy")

CLASSES = ("observe", "mutate", "execute")
_FIELDS = {"class", "confirm", "privileged", "blocked_channels"}


class PolicyError(ValueError):
    """policy.toml is malformed. Refuse to start rather than guess."""


def load_policy(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        log.warning("No policy file at %s; using the defaults declared in code", path)
        return {}
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    tools = data.get("tools", {})
    for name, entry in tools.items():
        unknown = set(entry) - _FIELDS
        if unknown:
            raise PolicyError(f"{path.name}: [tools.{name}] has unknown keys {sorted(unknown)}")
        if entry.get("class") not in CLASSES:
            raise PolicyError(f"{path.name}: [tools.{name}] class must be one of {CLASSES}")
    return tools


def apply_policy(registry: dict[str, ToolSpec], policy: dict[str, dict[str, Any]]) -> None:
    """Overwrite each tool's policy fields from the file."""
    for name in sorted(set(policy) - set(registry)):
        log.warning("policy.toml mentions unknown tool %r — ignored", name)
    for name, spec in registry.items():
        entry = policy.get(name)
        if entry is None:
            if policy:
                log.warning("policy.toml has no entry for %r; using its code defaults", name)
            continue
        spec.capability = entry["class"]
        spec.confirm = bool(entry.get("confirm", False))
        spec.privileged = bool(entry.get("privileged", False))
        spec.blocked_channels = frozenset(entry.get("blocked_channels", []))
