"""
Unit tests for G.I.D.E.O.N Telegram Bot Integration Daemon.
Tests access control security and message processing routing using standard asyncio.run.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from telegram_bot import GideonTelegramDaemon


def test_telegram_daemon_unauthorized_user():
    async def _test():
        daemon = GideonTelegramDaemon(token="FAKE_TOKEN", allowed_users={12345})

        mock_update = MagicMock()
        mock_update.effective_user.id = 99999  # Unauthorized user
        mock_update.message.reply_text = AsyncMock()

        await daemon._handle_message(mock_update, None)

        mock_update.message.reply_text.assert_called_once_with(
            "Access denied: Unauthorized Telegram User ID."
        )

    asyncio.run(_test())


def test_telegram_daemon_authorized_user_routing():
    async def _test():
        daemon = GideonTelegramDaemon(token="FAKE_TOKEN", allowed_users={12345})

        mock_update = MagicMock()
        mock_update.effective_user.id = 12345
        mock_update.effective_chat.id = 67890
        mock_update.message.text = "What is the weather in Paris?"
        mock_update.message.reply_text = AsyncMock()
        mock_update.message.reply_chat_action = AsyncMock()

        with patch("telegram_bot.run_chat_loop", return_value="The weather in Paris is 20°C."):
            await daemon._handle_message(mock_update, None)

        mock_update.message.reply_text.assert_called_once_with("The weather in Paris is 20°C.")

    asyncio.run(_test())
