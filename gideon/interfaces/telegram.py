"""
G.I.D.E.O.N Telegram Bot Integration Daemon

Runs an async Telegram bot worker tied to the engine's lifespan loop.
Allows remote messaging to G.I.D.E.O.N from mobile Telegram clients, with
strict user-ID authorization. Conversation goes through the same Agent as
every other interface, so memory is shared.

This process is the token's single getUpdates consumer; the always-on node
(node/) only ever sends.
"""

from __future__ import annotations

import asyncio
import random
from pathlib import Path
from typing import TYPE_CHECKING, Any

from gideon.core.config import (
    TELEGRAM_ALLOWED_USERS,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_SEND_VOICE,
)
from gideon.core.context import telegram_session_id
from gideon.core.logs import get_logger

if TYPE_CHECKING:
    from gideon.brain.agent import Agent

log = get_logger("gideon.telegram")

# Telegram rejects messages over 4096 characters.
_CHUNK = 4000
# Seconds between edits of a reply that is still being written. Telegram
# throttles a bot that edits one chat much faster than about once a second.
_EDIT_INTERVAL = 1.2
_NO_OUTPUT = "I processed your request, but produced no textual output."


class _StreamedReply:
    """A reply that grows in place: sent on the first words, then edited.

    Text past Telegram's length limit continues in a new message, so a long
    answer streams across several bubbles instead of failing at the end.
    """

    def __init__(self, message: Any) -> None:
        self._origin = message
        self._sent: list[Any] = []
        self._shown: list[str] = []

    async def show(self, text: str) -> None:
        chunks = [text[i : i + _CHUNK] for i in range(0, len(text), _CHUNK)] or [""]
        for index, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
            if index < len(self._sent):
                if self._shown[index] != chunk:
                    # Editing to identical text is an API error; skip it.
                    await self._sent[index].edit_text(chunk)
                    self._shown[index] = chunk
            else:
                self._sent.append(await self._origin.reply_text(chunk))
                self._shown.append(chunk)


class GideonTelegramDaemon:
    def __init__(
        self,
        agent: Agent | None = None,
        token: str = TELEGRAM_BOT_TOKEN,
        allowed_users: set[int] = TELEGRAM_ALLOWED_USERS,
        send_voice: bool = TELEGRAM_SEND_VOICE,
    ) -> None:
        self.agent = agent
        self.token = token.strip()
        self.allowed_users = allowed_users
        self.send_voice = send_voice
        self.app: Any = None

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
        session_id = telegram_session_id(chat_id)
        user_text = update.message.text.strip()

        # Send typing action
        try:
            await update.message.reply_chat_action("typing")
        except Exception:
            pass

        if self.agent is None:
            log.error("Telegram message received but no agent is wired up")
            return

        try:
            self.agent.db.ensure_session(session_id, f"Telegram Chat {chat_id}")
            self.agent.record_user_message(session_id, user_text)
            await self._stream_reply(update.message, session_id, user_text)
        except Exception as err:
            log.error("Error processing Telegram message from %s: %s", chat_id, err, exc_info=True)
            await update.message.reply_text(f"An error occurred while processing your request: {err}")

    async def _stream_reply(self, message: Any, session_id: str, user_text: str) -> None:
        """Generate on a worker thread and edit the reply in as it arrives.

        The agent loop blocks for the whole generation (several seconds);
        run on the event loop it would freeze SSE streams, the reminder
        daemon and the web UI. Tokens cross back to the loop through a queue.
        """
        agent = self.agent
        assert agent is not None
        loop = asyncio.get_running_loop()
        tokens: asyncio.Queue[str | None] = asyncio.Queue()

        def produce() -> None:
            try:
                for token in agent.stream(session_id, user_text, "telegram"):
                    loop.call_soon_threadsafe(tokens.put_nowait, token)
            finally:
                loop.call_soon_threadsafe(tokens.put_nowait, None)

        producer = loop.run_in_executor(None, produce)
        reply = _StreamedReply(message)
        text, last_edit = "", 0.0
        try:
            while (token := await tokens.get()) is not None:
                text += token
                if text.strip() and loop.time() - last_edit >= _EDIT_INTERVAL:
                    try:
                        await reply.show(text)
                    except Exception as exc:
                        # Interim edits are cosmetic; the final one below is not.
                        log.debug("Interim Telegram edit failed: %s", exc)
                    last_edit = loop.time()
        finally:
            await producer
        await reply.show(text if text.strip() else _NO_OUTPUT)

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

    async def send_message(self, chat_id: int | str, text: str) -> bool:
        """Push an unprompted message into a chat. Returns True on success."""
        if not self.app or not self.app.bot:
            log.debug("Telegram not running; dropping outbound message")
            return False
        try:
            for chunk_idx in range(0, len(text), _CHUNK):
                await self.app.bot.send_message(
                    chat_id=chat_id, text=text[chunk_idx : chunk_idx + _CHUNK]
                )
            return True
        except Exception as err:
            log.error("Failed to send Telegram message to %s: %s", chat_id, err)
            return False

    async def send_voice_note(self, chat_id: int | str, audio_path: str) -> bool:
        """Best-effort voice note. Telegram wants OGG/OPUS for sendVoice, and
        Piper produces WAV, so fall back to sendAudio then give up quietly."""
        if not self.app or not self.app.bot or not audio_path:
            return False
        path = Path(audio_path)
        if not path.is_file():
            return False
        for sender in ("send_voice", "send_audio"):
            try:
                with path.open("rb") as handle:
                    await getattr(self.app.bot, sender)(chat_id=chat_id, **{
                        "voice" if sender == "send_voice" else "audio": handle
                    })
                return True
            except Exception as err:
                log.debug("Telegram %s failed: %s", sender, err)
        log.info("Could not deliver voice note; text was still sent")
        return False

    async def broadcast(self, text: str, audio_path: str | None = None) -> int:
        """Send to every authorised user. Used when a reminder has no origin
        chat — one set from the desktop still needs to reach the phone."""
        sent = 0
        for user_id in self.allowed_users:
            if await self.send_message(user_id, text):
                sent += 1
                if self.send_voice and audio_path:
                    await self.send_voice_note(user_id, audio_path)
        return sent

    async def notify_reminder(
        self, text: str, chat_id: str | None = None, audio_path: str | None = None
    ) -> bool:
        """Deliver a fired reminder.

        Prefers the chat the reminder was created in; otherwise reaches every
        authorised user, so a reminder set on the desktop still arrives on the
        phone when no browser is open.
        """
        if not self.app:
            return False

        if chat_id:
            ok = await self.send_message(chat_id, text)
            if ok and self.send_voice and audio_path:
                await self.send_voice_note(chat_id, audio_path)
            return ok

        return await self.broadcast(text, audio_path) > 0

    async def stop(self) -> None:
        if self.app:
            try:
                log.info("Shutting down G.I.D.E.O.N Telegram Bot daemon...")
                # python-telegram-bot exposes `.running`, not `.is_running`.
                # With the wrong name this raised AttributeError before
                # stopping the updater, so polling was never shut down
                # cleanly — and the next start hit 409 Conflict because
                # Telegram still had the old getUpdates consumer registered.
                # That matters now that the engine restarts routinely (boot
                # task, tray "Restart Engine").
                if self.app.updater and self.app.updater.running:
                    await self.app.updater.stop()
                if self.app.running:
                    await self.app.stop()
                await self.app.shutdown()
                log.info("Telegram Bot daemon stopped.")
            except Exception as err:
                log.error("Error shutting down Telegram Bot daemon: %s", err)
