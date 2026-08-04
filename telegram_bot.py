"""
G.I.D.E.O.N Telegram Bot Integration Daemon

Runs an async Telegram bot worker tied to FastAPI's lifespan loop.
Allows remote messaging to G.I.D.E.O.N from mobile Telegram clients.
Includes strict user ID authorization and shared dual-tier SQLite + ChromaDB memory.
"""

from __future__ import annotations

import asyncio
from typing import Any

from config import (
    TELEGRAM_ALLOWED_USERS,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_SEND_VOICE,
    get_logger,
)
from db import DatabaseManager, get_db
from orchestrator import addMemoryAsync, run_chat_loop

log = get_logger("gideon.telegram")

_collection: Any = None


def _get_collection() -> Any:
    """Reuse one ChromaDB client. Building one per message is expensive and
    contends on the same persistent store."""
    global _collection
    if _collection is None:
        from memory import initMemory

        _collection = initMemory()
    return _collection


def _ensure_session(db: DatabaseManager, session_id: str, chat_id: Any) -> None:
    db.ensure_session(session_id, f"Telegram Chat {chat_id}")


def _record_user_message(db: DatabaseManager, collection: Any, session_id: str, text: str) -> None:
    """Persist the incoming message before generating.

    run_agent_stream builds its context from SQLite. If the prompt is not
    stored first, format_messages_for_ollama drops it and the model answers
    the previous turn instead.
    """
    from datetime import datetime

    message_id = db.insert_message(session_id, "user", text)
    addMemoryAsync(collection, message_id, text, session_id, "user", datetime.now().isoformat())


class GideonTelegramDaemon:
    def __init__(
        self,
        token: str = TELEGRAM_BOT_TOKEN,
        allowed_users: set[int] = TELEGRAM_ALLOWED_USERS,
    ) -> None:
        self.token = token.strip()
        self.allowed_users = allowed_users
        self.app: Any = None
        self._task: asyncio.Task | None = None

    def is_configured(self) -> bool:
        """A token alone is not enough — an empty allowlist is not a valid config.

        Telegram bot usernames are discoverable, and this bot can read the
        clipboard, read local documents, search the disk and launch apps. An
        unset allowlist must therefore fail closed, not open.
        """
        if not self.token:
            return False
        if not self.allowed_users:
            log.error(
                "GIDEON_TELEGRAM_BOT_TOKEN is set but GIDEON_TELEGRAM_ALLOWED_USERS "
                "is empty. Refusing to start the bot: an open bot would let any "
                "Telegram user run tools on this machine. Set your numeric user ID."
            )
            return False
        return True

    def _is_authorized(self, user_id: int | None) -> bool:
        """Deny by default. An empty allowlist authorises nobody."""
        if not self.allowed_users:
            return False
        return user_id in self.allowed_users

    async def _handle_start(self, update: Any, context: Any) -> None:
        user_id = update.effective_user.id if update.effective_user else None
        if not self._is_authorized(user_id):
            log.warning("Unauthorized Telegram connection attempt from User ID: %s", user_id)
            await update.message.reply_text("Access denied: Unauthorized Telegram User ID.")
            return

        import random

        greeting_title = random.choice(["boss", "sir"])
        await update.message.reply_text(
            f"Greetings! G.I.D.E.O.N online and synchronized. How may I assist you today {greeting_title}?"
        )

    async def _handle_message(self, update: Any, context: Any) -> None:
        if not update.message or not update.message.text:
            return

        user_id = update.effective_user.id if update.effective_user else None
        if not self._is_authorized(user_id):
            log.warning("Rejected message from unauthorized User ID: %s", user_id)
            await update.message.reply_text("Access denied: Unauthorized Telegram User ID.")
            return

        chat_id = update.effective_chat.id
        session_id = f"telegram_{chat_id}"
        user_text = update.message.text.strip()

        # Send typing action
        try:
            await update.message.reply_chat_action("typing")
        except Exception:
            pass

        db = get_db()
        chroma = _get_collection()
        _ensure_session(db, session_id, chat_id)
        _record_user_message(db, chroma, session_id, user_text)

        try:
            # run_chat_loop blocks for the whole generation (several seconds).
            # Called directly it would stall the shared event loop, freezing
            # SSE streams, the reminder daemon and the web UI.
            response_text = await asyncio.to_thread(
                run_chat_loop,
                db,
                chroma,
                session_id,
                user_text,
                "telegram",
            )

            if not response_text:
                response_text = "I processed your request, but produced no textual output."

            # Telegram max message length is 4096 chars
            if len(response_text) > 4000:
                for chunk_idx in range(0, len(response_text), 4000):
                    await update.message.reply_text(response_text[chunk_idx : chunk_idx + 4000])
            else:
                await update.message.reply_text(response_text)

        except Exception as err:
            log.error("Error processing Telegram message from %s: %s", chat_id, err, exc_info=True)
            await update.message.reply_text(f"An error occurred while processing your request: {err}")

    async def start(self) -> None:
        if not self.is_configured():
            log.info("Telegram bot token not provided (GIDEON_TELEGRAM_BOT_TOKEN empty). Skipping daemon.")
            return

        try:
            from telegram.ext import ApplicationBuilder, CommandHandler, MessageHandler, filters

            self.app = ApplicationBuilder().token(self.token).build()

            self.app.add_handler(CommandHandler("start", self._handle_start))
            self.app.add_handler(
                MessageHandler(filters.TEXT & ~filters.COMMAND, self._handle_message)
            )

            await self.app.initialize()
            await self.app.start()
            if self.app.updater:
                await self.app.updater.start_polling(drop_pending_updates=True)
            log.info("G.I.D.E.O.N Telegram Bot daemon started successfully.")
        except Exception as err:
            log.error("Failed to start Telegram Bot daemon: %s", err, exc_info=True)

    async def stop(self) -> None:
        if self.app:
            try:
                log.info("Shutting down G.I.D.E.O.N Telegram Bot daemon...")
                if self.app.updater and self.app.updater.is_running:
                    await self.app.updater.stop()
                if self.app.is_running:
                    await self.app.stop()
                await self.app.shutdown()
                log.info("Telegram Bot daemon stopped.")
            except Exception as err:
                log.error("Error shutting down Telegram Bot daemon: %s", err)
