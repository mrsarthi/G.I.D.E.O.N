"""
G.I.D.E.O.N Slot Validation — the "never assume" rule (PRD §5.5)

The model is never permitted to invent a required value. Before any tool
runs, the broker checks the call against the tool's declaration:

* **Missing** — a required parameter has no value → don't run; ask.
* **Ungrounded** — a parameter declared `grounded=True` holds something the
  user never said (not in this message, not in the recent conversation):
    - required → don't run; ask.
    - optional → drop the invented value; the tool's default applies
      (e.g. an invented city for the weather becomes "current location").

Grounding is a deliberately simple word-overlap test: at least one content
word of the value must appear in what the user said recently. It catches
the failures measured in F-12 — a reminder time the user never gave, a song
the user never named ("put on some Arctic Monkeys" → "Starboy by The
Weeknd") — while tolerating paraphrase ("F1" → "last Formula One race").
Values that legitimately transform the user's words (a ticker symbol from a
company name, a shell command from a description) are simply not declared
grounded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from gideon.tools.base import _MISSING_VALUES, Param, ToolSpec

_WORD = re.compile(r"[a-z0-9]+")

# Words too generic to prove a value came from the user.
_STOPWORDS = frozenset(
    ["a", "an", "the", "in", "at", "on", "to", "of", "for", "from", "by", "with", "and", "or", "is", "it", "this", "that", "my", "me", "your", "please", "now", "some", "any"]
)


def _stem(word: str) -> str:
    """Crude plural folding, so "reminders" matches "reminder"."""
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def _words(text: str) -> set[str]:
    return {_stem(w) for w in _WORD.findall(text.lower())}


def is_grounded(value: Any, grounding: str, ignore: tuple[str, ...] = ()) -> bool:
    """Whether `value` plausibly came from the user's own words."""
    ignored = {_stem(w) for w in _STOPWORDS | {w.lower() for w in ignore}}
    content = [w for w in _words(str(value)) if w not in ignored]
    if not content:
        # Nothing but filler ("the file") — it carries no information of its
        # own, so there is nothing to have invented. Missing-ness is checked
        # separately.
        return True
    return bool(set(content) & _words(grounding))


@dataclass
class SlotCheck:
    """Outcome of validating one tool call."""

    arguments: dict[str, Any]
    missing: list[Param] = field(default_factory=list)
    ungrounded: list[Param] = field(default_factory=list)
    dropped: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing and not self.ungrounded

    def problem(self) -> str:
        """A short label for traces, e.g. `missing:time_str`."""
        parts = [f"missing:{p.name}" for p in self.missing]
        parts += [f"ungrounded:{p.name}" for p in self.ungrounded]
        return ",".join(parts)

    def instruction(self) -> str:
        """What the model is told instead of a tool result."""
        needed = self.missing + self.ungrounded
        details = "; ".join(f"{p.name} ({p.description.rstrip('.')})" for p in needed)
        return (
            "Not run: the user has not told you "
            + ("this" if len(needed) == 1 else "these")
            + f" — {details}. Ask the user for it in one short question. "
            "Do not guess, do not invent a value, and do not say it is done."
        )


def check_call(spec: ToolSpec, arguments: dict[str, Any], grounding: str) -> SlotCheck:
    """Validate a model-requested call against its declaration."""
    args = dict(arguments)
    result = SlotCheck(arguments=args)
    for param in spec.params:
        present = [k for k in (param.name, *param.aliases) if args.get(k) not in _MISSING_VALUES]
        if not present:
            if param.required:
                result.missing.append(param)
            continue
        if not param.grounded:
            continue
        value = args[present[0]]
        if is_grounded(value, grounding, param.ground_ignore):
            continue
        if param.required:
            result.ungrounded.append(param)
        else:
            for key in present:
                args.pop(key, None)
            result.dropped.append(param.name)
    return result
