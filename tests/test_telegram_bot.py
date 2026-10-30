"""
Unit tests for G.I.D.E.O.N Telegram Bot Integration Daemon.
Tests access control security and message processing routing using standard asyncio.run.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock

from gideon.interfaces.telegram import GideonTelegramDaemon


def test_telegram_daemon_unauthorized_user():
    async def _test():
        agent = MagicMock()
        daemon = GideonTelegramDaemon(agent=agent, token="FAKE_TOKEN", allowed_users={12345})

        mock_update = MagicMock()
        mock_update.effective_user.id = 99999  # Unauthorized user
        mock_update.message.reply_text = AsyncMock()

        await daemon._handle_message(mock_update, None)

        mock_update.message.reply_text.assert_called_once_with(
            "Access denied: Unauthorized Telegram User ID."
        )
        agent.respond.assert_not_called()

    asyncio.run(_test())


def _update(text="What is the weather in Paris?"):
    update = MagicMock()
    update.effective_user.id = 12345
    update.effective_chat.id = 67890
    update.message.text = text
    sent = MagicMock()
    sent.edit_text = AsyncMock()
    update.message.reply_text = AsyncMock(return_value=sent)
    update.message.reply_chat_action = AsyncMock()
    return update, sent


def test_telegram_daemon_authorized_user_routing():
    async def _test():
        agent = MagicMock()
        agent.stream.return_value = iter(["The weather ", "in Paris is 20°C."])
        daemon = GideonTelegramDaemon(agent=agent, token="FAKE_TOKEN", allowed_users={12345})
        update, sent = _update()

        await daemon._handle_message(update, None)

        # Sent on the first words, then edited to the full reply.
        update.message.reply_text.assert_called_once_with("The weather ")
        sent.edit_text.assert_awaited_with("The weather in Paris is 20°C.")
        # Recorded before generating, on the chat's own session, over the
        # telegram channel (which withholds shell and app-launch tools).
        agent.record_user_message.assert_called_once_with(
            "telegram_67890", "What is the weather in Paris?"
        )
        agent.stream.assert_called_once_with(
            "telegram_67890", "What is the weather in Paris?", "telegram"
        )

    asyncio.run(_test())


def test_long_replies_continue_in_a_new_message():
    """Telegram caps a message at 4096 characters; the stream must not fail
    at the end of a long answer."""
    async def _test():
        agent = MagicMock()
        agent.stream.return_value = iter(["a" * 3000, "b" * 3000])
        daemon = GideonTelegramDaemon(agent=agent, token="FAKE_TOKEN", allowed_users={12345})
        update, _ = _update()

        await daemon._handle_message(update, None)

        texts = [c.args[0] for c in update.message.reply_text.await_args_list]
        assert len(texts) == 2
        assert all(len(t) <= 4000 for t in texts)

    asyncio.run(_test())


def test_empty_reply_still_answers():
    async def _test():
        agent = MagicMock()
        agent.stream.return_value = iter([])
        daemon = GideonTelegramDaemon(agent=agent, token="FAKE_TOKEN", allowed_users={12345})
        update, _ = _update()

        await daemon._handle_message(update, None)

        assert "no textual output" in update.message.reply_text.await_args.args[0]

    asyncio.run(_test())
