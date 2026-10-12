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


# Pointing words: "copy that" names nothing the model could copy.
_POINTERS = frozenset(["that", "this", "it", "these", "those", "them", "something", "stuff"])


def is_informative(value: Any) -> bool:
    """Whether `value` says anything beyond filler and pointing words.

    Not the tool's `ground_ignore` words: those are poor *evidence* but can be
    real values (search_web(query="news")).
    """
    ignored = {_stem(w) for w in _STOPWORDS | _POINTERS}
    return any(w not in ignored for w in _words(str(value)))


def is_grounded(value: Any, grounding: str, ignore: tuple[str, ...] = ()) -> bool:
    """Whether `value` plausibly came from the user's own words."""
    # Pointing words ("copy that") are no evidence of where a value came from.
    ignored = {_stem(w) for w in _STOPWORDS | _POINTERS | {w.lower() for w in ignore}}
    content = [w for w in _words(str(value)) if w not in ignored]
    if not content:
        # Nothing but filler ("the file") — it carries no information of its
        # own, so there is nothing to have invented. Missing-ness is checked
        # separately.
        return True
    said = _words(grounding)
    # Two words run together, either way round: the user said "vs code" and
    # the model wrote "vscode", or the user typed "vscode" and the model
    # wrote "vs code".
    # Only words that count as evidence pair up: "copy that" run together is
    # still no evidence.
    said |= _pairs(_WORD.findall(grounding.lower()), ignored)
    content += _pairs(_WORD.findall(str(value).lower()), ignored)
    return any(_same_word(word, heard) for word in content for heard in said)


def _pairs(words: list[str], ignored: set[str]) -> set[str]:
    return {
        a + b
        for a, b in zip(words, words[1:], strict=False)
        if _stem(a) not in ignored and _stem(b) not in ignored
    }


def _same_word(value_word: str, said_word: str) -> bool:
    """Exact, or one is a shortening of the other: the user said "calculator"
    and the model wrote "calc" (seen live), or said "spot" for "spotify"."""
    if value_word == said_word:
        return True
    shorter, longer = sorted((value_word, said_word), key=len)
    return len(shorter) >= 4 and longer.startswith(shorter)


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
        if not param.grounded and param.evidence is None:
            continue
        value = args[present[0]]
        if param.required and not is_informative(value):
            # set_clipboard(text="That") for "copy that to my clipboard":
            # the value is present but says nothing, so it is still missing.
            result.missing.append(param)
            continue
        if param.evidence is not None:
            said = _words(grounding)
            if said & {_stem(w) for w in param.evidence.get(str(value).lower(), ())}:
                continue
        elif is_grounded(value, grounding, param.ground_ignore):
            continue
        if param.required:
            result.ungrounded.append(param)
        else:
            for key in present:
                args.pop(key, None)
            result.dropped.append(param.name)
    return result
