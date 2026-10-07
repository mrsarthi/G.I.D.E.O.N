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


def test_telegram_daemon_authorized_user_routing():
    async def _test():
        agent = MagicMock()
        agent.respond.return_value = "The weather in Paris is 20°C."
        daemon = GideonTelegramDaemon(agent=agent, token="FAKE_TOKEN", allowed_users={12345})

        mock_update = MagicMock()
        mock_update.effective_user.id = 12345
        mock_update.effective_chat.id = 67890
        mock_update.message.text = "What is the weather in Paris?"
        mock_update.message.reply_text = AsyncMock()
        mock_update.message.reply_chat_action = AsyncMock()

        await daemon._handle_message(mock_update, None)

        mock_update.message.reply_text.assert_called_once_with("The weather in Paris is 20°C.")
        # Recorded before generating, on the chat's own session, over the
        # telegram channel (which withholds shell and app-launch tools).
        agent.record_user_message.assert_called_once_with(
            "telegram_67890", "What is the weather in Paris?"
        )
        agent.respond.assert_called_once_with(
            "telegram_67890", "What is the weather in Paris?", "telegram"
        )

    asyncio.run(_test())
