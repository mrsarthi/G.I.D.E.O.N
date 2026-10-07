"""
Move runtime state from the project root into data/.

    .venv\\Scripts\\python.exe scripts\\migrate_data_dir.py          # dry run
    .venv\\Scripts\\python.exe scripts\\migrate_data_dir.py --apply

Stop the engine and the tray first (tray menu -> Quit). SQLite holds the
database open with a WAL, and a half-moved database looks exactly like data
loss — which is why this never happens implicitly at startup. Until it runs,
config.py keeps using the legacy root paths, so nothing breaks in between.
"""

from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# (legacy path relative to ROOT, new path relative to DATA)
MOVES = [
    ("gideon.db", "gideon.db"),
    ("gideon.db-wal", "gideon.db-wal"),
    ("gideon.db-shm", "gideon.db-shm"),
    (".chromadb", "chroma"),
    ("logs", "logs"),
    (".webview", "webview"),
]


def _checkpoint(db_path: Path) -> None:
    """Fold the WAL into the main file so the database moves as one file."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--apply", action="store_true", help="actually move the files")
    args = parser.parse_args()

    planned = []
    for legacy_name, new_name in MOVES:
        legacy, target = ROOT / legacy_name, DATA / new_name
        if not legacy.exists():
            continue
        if target.exists():
            print(f"skip   {legacy_name}: {target} already exists")
            continue
        planned.append((legacy, target))

    if not planned:
        print("Nothing to migrate.")
        return 0

    for legacy, target in planned:
        print(f"{'move' if args.apply else 'would move'}  {legacy} -> {target}")

    if not args.apply:
        print("\nDry run. Stop the engine and tray, then re-run with --apply.")
        return 0

    db = ROOT / "gideon.db"
    if db.exists():
        try:
            _checkpoint(db)
        except sqlite3.Error as exc:
            print(f"Could not checkpoint {db} ({exc}). Is the engine still running?")
            return 1

    DATA.mkdir(parents=True, exist_ok=True)
    for legacy, target in planned:
        if not legacy.exists():  # the checkpoint can remove -wal / -shm
            continue
        try:
            shutil.move(str(legacy), str(target))
        except OSError as exc:
            print(f"Failed to move {legacy}: {exc}. Is the engine or tray still running?")
            return 1

    print("Done. Start G.I.D.E.O.N as usual.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
