"""Test-suite isolation.

Runs before any gideon module is imported, so configuration resolves against
a throwaway data directory. Without this the suite wrote into the real
gideon.db — including pending reminders that would later fire on Telegram —
and could reach a configured remote scheduler or Telegram bot.
"""

from __future__ import annotations

import os
import tempfile

# Reused if this module is imported more than once in a session — otherwise
# each import made its own folder and only the last one was cleaned up.
_DATA_DIR = os.environ.get("_GIDEON_TEST_DATA_DIR") or tempfile.mkdtemp(prefix="gideon-tests-")
os.environ["_GIDEON_TEST_DATA_DIR"] = _DATA_DIR

os.environ["GIDEON_DATA_DIR"] = _DATA_DIR
# Empty values win over .env, which only fills keys that are unset.
os.environ["GIDEON_REMOTE_SCHEDULER_URL"] = ""
os.environ["GIDEON_REMOTE_SCHEDULER_TOKEN"] = ""
os.environ["GIDEON_TELEGRAM_BOT_TOKEN"] = ""
# Results must not depend on whether the laptop is plugged in while tests run.
os.environ["GIDEON_ON_BATTERY"] = "cpu"
# Never load the real model as a side effect of starting the app in a test.
os.environ["GIDEON_WARM_UP"] = "0"


def pytest_sessionfinish(session, exitstatus):
    """Delete the throwaway data directory. Without this every run left a
    `gideon-tests-*` folder behind in Temp (30 had piled up)."""
    import shutil

    try:
        from gideon.storage.db import close_db

        close_db()  # Windows will not delete a database file that is still open
    except Exception:
        pass
    shutil.rmtree(_DATA_DIR, ignore_errors=True)
