"""
G.I.D.E.O.N Reply Style

At most one "boss" or "sir" per reply, in its first sentence.

The system prompt asks for exactly that (RULE 7), and the live context used
to repeat it every turn. A 4B model ignores it: of 300 stored replies
(2026-10-27), 53 addressed the user two or more times and 103 ended with
"boss" or "sir" — "Boss, the calculator is open, sir." The rule is enforced
here instead, on the text as it streams, so what is shown, spoken and stored
all agree.

Only addresses are touched: "my boss", "the final boss" and "Sir Isaac
Newton" are not addresses and stay.
"""

from __future__ import annotations

import re

# "boss"/"sir" as a word, with the comma or spaces that lead into it.
_CANDIDATE = re.compile(r"(?P<lead>,[ \t]*|[ \t]+)?\b(?P<word>boss|sir)\b", re.IGNORECASE)
# Words before "boss"/"sir" that make it something other than an address.
_NOT_ADDRESS_BEFORE = re.compile(
    r"\b(?:my|your|his|her|their|our|the|a|an|this|that|final|big|real|mini|level|end|new|old|own)\s*$",
    re.IGNORECASE,
)
# A title before a name ("Sir Isaac Newton") — but "boss I'll" is an address.
_TITLE_BEFORE_NAME = re.compile(r" (?!I\b)[A-Z]")
_SENTENCE_END = re.compile(r"[.!?](?:\s|$)")
_AT_SENTENCE_START = re.compile(r"(?:^|[.!?])\s*$")
# Enough text held back to see a whole ", sir" or " boss" and what follows.
_HOLD = 16


def _is_address(before_word: str, after_word: str) -> bool:
    return not _NOT_ADDRESS_BEFORE.search(before_word) and not _TITLE_BEFORE_NAME.match(after_word)


class AddressLimiter:
    """Streams a reply through, keeping only its first address.

    feed() returns the text that can be shown now; flush() returns the rest
    at the end. A short tail is held back so ", sir" can be judged whole.
    """

    def __init__(self) -> None:
        self._pending = ""
        self._emitted = ""
        self._kept = False
        self._capitalize = False

    def feed(self, text: str) -> str:
        self._pending += text
        return self._drain(final=False)

    def flush(self) -> str:
        return self._drain(final=True)

    def _drain(self, final: bool) -> str:
        buf = self._pending
        limit = len(buf) if final else max(0, len(buf) - _HOLD)
        out = ""
        pos = 0
        for match in _CANDIDATE.finditer(buf):
            if match.start() >= limit:
                break
            if not final and len(buf) - match.end() < 2:
                # Can't see what follows the word yet ("sir" vs "Sir Isaac").
                limit = match.start()
                break
            out += self._text(buf[pos:match.start()])
            before = self._emitted + out
            lead = match.group("lead") or ""
            pos = match.end()
            if not _is_address(before + lead, buf[pos:]):
                out += self._text(match.group(0))
            elif not self._kept and not _SENTENCE_END.search(before):
                self._kept = True  # the one address: in the first sentence
                out += self._text(match.group(0))
            elif "," not in lead and _AT_SENTENCE_START.search(before):
                # "… open. Sir, anything else?" → "… open. Anything else?"
                if buf[pos:pos + 1] == ",":
                    pos += 1
                while buf[pos:pos + 1] in (" ", "\t"):
                    pos += 1
                out += lead if before and not before[-1].isspace() else ""
                self._capitalize = True
            # else: ", sir" / " boss" mid-sentence — dropped with its lead.
        end = max(limit, pos)
        out += self._text(buf[pos:end])
        self._pending = buf[end:]
        self._emitted += out
        return out

    def _text(self, segment: str) -> str:
        """Kept text, capitalising the first letter after a removed "Sir, "."""
        if not self._capitalize or not segment:
            return segment
        for i, ch in enumerate(segment):
            if ch.isalpha():
                self._capitalize = False
                return segment[:i] + ch.upper() + segment[i + 1:]
            if not ch.isspace():
                self._capitalize = False
                return segment
        return segment


def limit_addresses(text: str) -> str:
    """The whole-text form of AddressLimiter, for stored replies."""
    limiter = AddressLimiter()
    return limiter.feed(text) + limiter.flush()
