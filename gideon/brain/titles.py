"""
G.I.D.E.O.N Session Titles

A session created without a name gets a placeholder and a flag; after its
first exchange the model names it in a few words, so the HUD's session list
reads "GPU temperature check" instead of a column of timestamps.

Runs on one background thread after the reply has finished streaming, so it
never delays speech. The request reuses the chat model's exact num_ctx:
Ollama reloads a model whose context size changes, which on a 4GB card would
cost seconds and evict whatever else was resident.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import ollama

from gideon.brain.llm import chat_options, model_paused, thinking_kwargs
from gideon.core.config import CHAT_KEEP_ALIVE, CHAT_MODEL
from gideon.core.logs import get_logger
from gideon.storage.db import DatabaseManager

log = get_logger("gideon.brain.titles")

PLACEHOLDER_TITLE = "New conversation"
MAX_TITLE_CHARS = 48

_PROMPT = (
    "Write a title of 2 to 5 words for a conversation that starts like this. "
    "Reply with the title only: no quotes, no punctuation at the end.\n\n"
    "User: {user}\nAssistant: {reply}"
)

Publish = Callable[[str, dict[str, Any]], None]


def clean_title(raw: str) -> str:
    """First line, unquoted, no trailing punctuation, bounded length."""
    line = (raw or "").strip().splitlines()[0] if (raw or "").strip() else ""
    line = re.sub(r"^(title\s*:\s*)", "", line, flags=re.IGNORECASE)
    line = line.strip().strip("\"'`*#").strip().rstrip(".!?:;,")
    if len(line) > MAX_TITLE_CHARS:
        line = line[:MAX_TITLE_CHARS].rsplit(" ", 1)[0]
    return line


def fallback_title(user_text: str) -> str:
    """The first few words of the user's message, when the model is unavailable."""
    words = re.findall(r"[\w'’-]+", user_text)[:6]
    return clean_title(" ".join(words).capitalize()) or PLACEHOLDER_TITLE


class SessionTitler:
    def __init__(
        self,
        db: DatabaseManager,
        publish: Publish | None = None,
        chat: Callable[..., Any] = ollama.chat,
    ) -> None:
        self.db = db
        self.publish = publish
        self._chat = chat
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="session_titler")

    def maybe_title(self, session_id: str, user_text: str, reply: str) -> None:
        """Queue a title for this session if it still has a placeholder name."""
        if user_text.strip() and self.db.session_wants_title(session_id):
            self._executor.submit(self._title, session_id, user_text, reply)

    def _generate(self, user_text: str, reply: str) -> str:
        response = self._chat(
            model=CHAT_MODEL,
            messages=[
                {"role": "user", "content": _PROMPT.format(user=user_text[:500], reply=reply[:500])}
            ],
            stream=False,
            options=chat_options(num_predict=16, temperature=0.2),
            keep_alive=CHAT_KEEP_ALIVE,
            **thinking_kwargs(CHAT_MODEL),
        )
        message = getattr(response, "message", None) or (
            response.get("message") if isinstance(response, dict) else None
        )
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        return clean_title(content or "")

    def _title(self, session_id: str, user_text: str, reply: str) -> None:
        try:
            title = "" if model_paused() else self._generate(user_text, reply)
        except Exception as exc:
            log.debug("Title generation failed (%s); using the first words", exc)
            title = ""
        title = title or fallback_title(user_text)

        # Re-check: the user may have renamed it while we were generating.
        if not self.db.session_wants_title(session_id):
            return
        self.db.rename_session(session_id, title)
        log.info("Titled session %s: %r", session_id, title)
        if self.publish is not None:
            self.publish("session_renamed", {"session_id": session_id, "session_name": title})

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
