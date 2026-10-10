"""
G.I.D.E.O.N Traces

One structured record per agent turn (PRD §6.3):

    request → tools offered → tool calls (with slots) → outcome → reply
           → latency per stage → the user's correction, when it comes

Traces are the eval suite's raw material, the regression net for model swaps,
and — eventually — training data. They are written for every channel (HUD,
quick window, Telegram, terminal) because the agent loop is shared.

Corrections arrive later and from different places:

* an explicit rating (👍/👎 in the HUD)                      → `rate()`
* a denied approval for an action the turn requested        → brain/approvals.py
* the user's next message reads like a correction          → `note_user_message()`
  ("no, I meant…", "that's wrong", "actually…") within a few minutes

Recording must never break a conversation: every write is best-effort.
"""

from __future__ import annotations

import re
import time
import uuid
from datetime import datetime, timedelta
from typing import Any

from gideon.core.logs import get_logger
from gideon.storage.db import DatabaseManager

log = get_logger("gideon.brain.traces")

# How long after a reply a correcting message still refers to it.
CORRECTION_WINDOW = timedelta(minutes=5)
_RESULT_EXCERPT_CHARS = 300

_CORRECTION = re.compile(
    r"^\s*(?:no\b|nope\b|wrong\b|incorrect\b|not\s+(?:that|what)\b|that'?s\s+(?:not|wrong)|"
    r"that\s+is\s+(?:not|wrong)|i\s+meant\b|i\s+said\b|actually\b|stop\b|cancel\s+that\b)",
    re.IGNORECASE,
)


def looks_like_correction(text: str) -> bool:
    return bool(_CORRECTION.match(text or ""))


def ollama_stats(chunk: Any) -> dict[str, Any]:
    """Timing counters Ollama attaches to the final chunk of a stream."""
    if not getattr(chunk, "done", False):
        return {}

    def ms(field: str) -> int | None:
        value = getattr(chunk, field, None)
        return round(value / 1_000_000) if value else None

    stats = {
        "load_ms": ms("load_duration"),
        "prompt_tokens": getattr(chunk, "prompt_eval_count", None),
        "prompt_ms": ms("prompt_eval_duration"),
        "output_tokens": getattr(chunk, "eval_count", None),
        "output_ms": ms("eval_duration"),
    }
    if stats["output_tokens"] and stats["output_ms"]:
        stats["tokens_per_sec"] = round(stats["output_tokens"] / (stats["output_ms"] / 1000), 1)
    return {k: v for k, v in stats.items() if v is not None}


class TraceRecorder:
    """Accumulates one turn's trace while the agent runs it."""

    def __init__(self, session_id: str, channel: str, request: str, model: str) -> None:
        self.trace_id = uuid.uuid4().hex
        self.session_id = session_id
        self.channel = channel
        self.request = request
        self.model = model
        self.created_at = datetime.now().isoformat(timespec="milliseconds")
        self.placement: str | None = None
        self.tools_offered = False
        self.tools_offered_names: list[str] = []
        self.outcome = "aborted"  # overwritten unless the stream is cut short
        self.reply = ""
        self.error: str | None = None
        self.passes: list[dict[str, Any]] = []
        self.stages: dict[str, int] = {}
        self.first_text_ms: int | None = None
        self._t0 = time.perf_counter()

    # ------------------------------------------------------------------
    def elapsed_ms(self) -> int:
        return round((time.perf_counter() - self._t0) * 1000)

    def stage(self, name: str) -> None:
        """Record when a named stage finished, relative to the turn's start."""
        self.stages[name] = self.elapsed_ms()

    def text_emitted(self, text: str) -> None:
        if self.first_text_ms is None and text.strip():
            self.first_text_ms = self.elapsed_ms()
        self.reply += text

    def start_pass(self) -> dict[str, Any]:
        record: dict[str, Any] = {"started_ms": self.elapsed_ms(), "tool_calls": []}
        self.passes.append(record)
        return record

    @staticmethod
    def first_chunk(record: dict[str, Any], started_ms: int, now_ms: int) -> None:
        record.setdefault("first_chunk_ms", now_ms - started_ms)

    def end_pass(self, record: dict[str, Any], stats: dict[str, Any]) -> None:
        record["llm_ms"] = self.elapsed_ms() - record["started_ms"]
        record.update(stats)

    @staticmethod
    def tool_call(
        record: dict[str, Any],
        *,
        name: str,
        args: dict[str, Any],
        source: str,
        corrected_from: str | None,
        approval: dict[str, Any] | None,
        result: str,
        ms: int,
    ) -> None:
        record["tool_calls"].append(
            {
                "name": name,
                "args": args,
                "source": source,
                "corrected_from": corrected_from,
                "approval": approval,
                "result": (result or "")[:_RESULT_EXCERPT_CHARS],
                "ms": ms,
            }
        )

    @property
    def tool_calls(self) -> list[dict[str, Any]]:
        return [call for p in self.passes for call in p["tool_calls"]]

    def to_record(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "channel": self.channel,
            "created_at": self.created_at,
            "request": self.request,
            "model": self.model,
            "placement": self.placement,
            "tools_offered": self.tools_offered,
            "outcome": self.outcome,
            "reply": self.reply,
            "error": self.error,
            "total_ms": self.elapsed_ms(),
            "first_text_ms": self.first_text_ms,
            "detail": {
                "tools_offered_names": self.tools_offered_names,
                "stages": self.stages,
                "passes": self.passes,
            },
        }


class TraceStore:
    """Persists traces and attaches corrections to them."""

    def __init__(self, db: DatabaseManager) -> None:
        self.db = db

    def save(self, recorder: TraceRecorder) -> None:
        try:
            self.db.insert_trace(recorder.to_record())
        except Exception as exc:
            log.warning("Could not save trace %s: %s", recorder.trace_id, exc)

    def annotate(self, trace_id: str, **fields: Any) -> None:
        """Merge interface-side measurements (e.g. first audio) into a saved trace."""
        try:
            self.db.annotate_trace(trace_id, fields)
        except Exception as exc:
            log.debug("Could not annotate trace %s: %s", trace_id, exc)

    def rate(self, trace_id: str, rating: int, note: str | None = None) -> bool:
        rating = 1 if rating > 0 else -1
        return self.db.record_correction(
            trace_id, kind="rating" if note else None, text=note or None, rating=rating
        )

    def note_user_message(self, session_id: str, text: str) -> None:
        """If this message corrects the previous turn, record it on that turn."""
        if not looks_like_correction(text):
            return
        try:
            previous = self.db.latest_trace(session_id)
            if previous is None or previous.get("correction_kind"):
                return
            created = datetime.fromisoformat(previous["created_at"])
            if datetime.now() - created > CORRECTION_WINDOW:
                return
            self.db.record_correction(previous["trace_id"], kind="rephrase", text=text)
            log.info("Recorded correction on trace %s", previous["trace_id"])
        except Exception as exc:
            log.debug("Could not check for a correction: %s", exc)
