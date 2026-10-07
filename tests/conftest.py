"""Test-suite isolation.

Runs before any gideon module is imported, so configuration resolves against
a throwaway data directory. Without this the suite wrote into the real
gideon.db — including pending reminders that would later fire on Telegram —
and could reach a configured remote scheduler or Telegram bot.
"""

from __future__ import annotations

import os
import tempfile

_DATA_DIR = tempfile.mkdtemp(prefix="gideon-tests-")

os.environ["GIDEON_DATA_DIR"] = _DATA_DIR
# Empty values win over .env, which only fills keys that are unset.
os.environ["GIDEON_REMOTE_SCHEDULER_URL"] = ""
os.environ["GIDEON_REMOTE_SCHEDULER_TOKEN"] = ""
os.environ["GIDEON_TELEGRAM_BOT_TOKEN"] = ""
