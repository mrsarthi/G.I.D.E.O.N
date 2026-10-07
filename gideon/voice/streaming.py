"""
G.I.D.E.O.N Streaming Speech

Speech is synthesised per sentence while the model is still generating, so
G.I.D.E.O.N starts talking after the first sentence instead of after the
whole answer. Measured: first audio at 8.96s on a reply that finished at
12.4s.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from concurrent.futures import Future
from typing import Any

from gideon.core.config import MIN_TTS_SENTENCE_CHARS
from gideon.core.logs import get_logger
from gideon.voice.tts import SpeechSynthesizer

log = get_logger("gideon.voice.streaming")

# Sentence boundary: terminator followed by whitespace.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…])\s+")


def split_ready_sentences(buffer: str) -> tuple[list[str], str]:
    """Split a streaming buffer into complete sentences plus a remainder.

    Short fragments are merged forward so a stray "Yes." does not become its
    own audio file.
    """
    parts = _SENTENCE_SPLIT.split(buffer)
    if len(parts) < 2:
        return [], buffer

    *complete, remainder = parts
    sentences: list[str] = []
    carry = ""
    for part in complete:
        candidate = f"{carry} {part}".strip() if carry else part
        if len(candidate) < MIN_TTS_SENTENCE_CHARS:
            carry = candidate
            continue
        sentences.append(candidate)
        carry = ""

    if carry:
        remainder = f"{carry} {remainder}".strip()
    return sentences, remainder


def speak_as_you_go(
    tokens: Iterable[str],
    synthesizer: SpeechSynthesizer,
    session_id: str,
    speak: bool = True,
) -> Iterator[dict[str, Any]]:
    """Interleave text deltas with audio URLs as each sentence is synthesised.

    Yields {"text": ...} for every token and {"audio_url": ...} for each
    finished utterance, in submission order. Synthesis runs concurrently with
    continued generation; audio is emitted as soon as it is ready without
    letting a later sentence overtake an earlier one.
    """
    buffer = ""
    pending: list[Future] = []

    def drain(block: bool) -> Iterator[dict[str, Any]]:
        while pending:
            future = pending[0]
            if not block and not future.done():
                break
            try:
                url = future.result(timeout=30 if block else None)
            except Exception as exc:
                log.error("Speech synthesis failed: %s", exc)
                url = ""
            pending.pop(0)
            if url:
                yield {"audio_url": url}

    for token in tokens:
        yield {"text": token}
        if not speak:
            continue

        buffer += token
        sentences, buffer = split_ready_sentences(buffer)
        for sentence in sentences:
            pending.append(synthesizer.submit(sentence, session_id))
        yield from drain(block=False)

    tail = buffer.strip()
    if speak and tail:
        pending.append(synthesizer.submit(tail, session_id))
    yield from drain(block=True)
