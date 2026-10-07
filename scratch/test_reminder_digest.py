"""
Manual check: the missed-reminder grace window and digest.

Inserts one reminder that is barely late and several that are badly late, then
waits for the running engine's daemon to pick them up. Expected in
the engine log (data/logs/engine.log, or logs/ before migrating) within ~35s (one poll cycle):

    Fired reminder #N (origin=web, telegram=sent)
    Retired 3 overdue reminder(s): 0 delivered by the node, 3 missed,
                                   2 recent enough to report

Before the grace window existed, all four would have fired individually, each
with its own synthesised speech and its own Telegram message.

    .venv\\Scripts\\python.exe scratch\\test_reminder_digest.py

NOTE: this sends real Telegram messages — one reminder and one digest. It
cleans up its own rows afterwards.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gideon.storage.db import get_db  # noqa: E402

POLL_WAIT = 40  # one reminder-daemon cycle plus slack


def main() -> int:
    db = get_db()
    now = datetime.now()

    cases = [
        ("A: 2 min late — should FIRE normally", timedelta(minutes=2)),
        ("B: 3 hours late — should be MISSED and reported", timedelta(hours=3)),
        ("C: 30 hours late — should be MISSED and reported", timedelta(hours=30)),
        ("D: 40 days late — MISSED but too old to report", timedelta(days=40)),
    ]

    ids = []
    for text, delta in cases:
        rid = db.add_reminder(text, (now - delta).isoformat(timespec="seconds"))
        ids.append(rid)
        print(f"  #{rid}  {text}")

    print(f"\nWaiting {POLL_WAIT}s for the reminder daemon...\n")
    time.sleep(POLL_WAIT)

    print("Resulting status (expect: 1 completed, 3 missed)")
    placeholders = ",".join("?" * len(ids))
    rows = db.conn.execute(
        f"SELECT reminder_id, status, text FROM reminders "
        f"WHERE reminder_id IN ({placeholders}) ORDER BY reminder_id",
        ids,
    ).fetchall()
    for rid, status, text in rows:
        print(f"  #{rid}  {status:<10} {text}")

    with db._lock, db.conn:
        db.conn.execute(
            f"DELETE FROM reminders WHERE reminder_id IN ({placeholders})", ids
        )
    print("\nTest rows removed. Now check the engine log (data/logs/engine.log, or logs/ before migrating) for the digest line.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
