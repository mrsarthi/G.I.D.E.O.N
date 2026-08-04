"""
G.I.D.E.O.N Relational Store (SQLite)

The connection is shared across the FastAPI worker threads, the ChromaDB
writer thread and the reminder daemon, so every statement goes through one
re-entrant lock. sqlite3 with check_same_thread=False is not thread-safe on
its own; concurrent writes without the lock surface as
"database table is locked" under load.

Use get_db() rather than constructing DatabaseManager per call site — each
instance opens another connection to the same WAL file.
"""

from __future__ import annotations

import sqlite3
import threading
import uuid
from datetime import datetime

from config import DB_PATH, get_logger

log = get_logger("gideon.db")

_SCHEMA_ROLES = ("user", "assistant", "tool", "tool_call", "system")


class DatabaseManager:
    def __init__(self, db_path=None):
        self.db_path = str(db_path or DB_PATH)
        self._lock = threading.RLock()
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)

        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        # Wait rather than fail immediately when another thread holds a write.
        self.conn.execute("PRAGMA busy_timeout=5000")

        self._init_schema()
        self._migrate()

    # ------------------------------------------------------------------
    #  Schema
    # ------------------------------------------------------------------
    def _init_schema(self):
        with self._lock, self.conn:
            cursor = self.conn.cursor()

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session (
                    session_id TEXT PRIMARY KEY,
                    session_name TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    tool_name TEXT DEFAULT 'none',
                    tool_args TEXT,
                    quality_flag INTEGER DEFAULT 0,
                    FOREIGN KEY (session_id) REFERENCES session(session_id) ON DELETE CASCADE
                )
            """)

            # Covers the sliding-window query (session filter + id ordering).
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session_id
                ON messages(session_id, message_id DESC)
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS user_profile (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reminders (
                    reminder_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text TEXT NOT NULL,
                    remind_at TEXT NOT NULL,
                    status TEXT DEFAULT 'pending',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_reminders_status_time
                ON reminders(status, remind_at)
            """)

            defaults = [
                ("user_name", "boss"),
                ("addressing_style", "boss or sir"),
                ("assistant_name", "G.I.D.E.O.N"),
                ("language_preference", "English"),
                ("ongoing_projects", "G.I.D.E.O.N AI Assistant Development"),
            ]
            for k, v in defaults:
                cursor.execute(
                    "INSERT OR IGNORE INTO user_profile (key, value) VALUES (?, ?)",
                    (k, v),
                )

    def _migrate(self):
        """Reclassify tool bookkeeping that older builds stored as assistant speech.

        Left as 'assistant' it gets replayed as conversation, which teaches the
        model to narrate "[Tool Call: ...]" instead of speaking naturally.
        """
        with self._lock, self.conn:
            cursor = self.conn.cursor()
            cursor.execute(
                "UPDATE messages SET role = 'tool_call' "
                "WHERE role = 'assistant' AND content LIKE '[Tool Call:%'"
            )
            if cursor.rowcount:
                log.info("Migrated %d legacy tool-call rows", cursor.rowcount)

    # ------------------------------------------------------------------
    #  Sessions & messages
    # ------------------------------------------------------------------
    def create_session(self, session_name="New Conversation") -> str:
        session_id = str(uuid.uuid4())
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO session (session_id, session_name) VALUES (?, ?)",
                (session_id, session_name),
            )
        return session_id

    def ensure_session(self, session_id: str, session_name: str = "Session") -> str:
        """Create a session with a caller-supplied id if it does not exist.

        Needed for external channels (Telegram) that derive a stable session id
        from the chat id. Messages carry a foreign key to session, so inserting
        one for an unknown session raises IntegrityError.
        """
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT OR IGNORE INTO session (session_id, session_name) VALUES (?, ?)",
                (session_id, session_name),
            )
        return session_id

    def session_exists(self, session_id: str) -> bool:
        with self._lock:
            row = self.conn.execute(
                "SELECT 1 FROM session WHERE session_id = ?", (session_id,)
            ).fetchone()
        return row is not None

    def list_sessions(self) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                "SELECT session_id, session_name, created_at FROM session "
                "ORDER BY created_at DESC"
            ).fetchall()
        return [
            {"session_id": r[0], "session_name": r[1], "created_at": r[2]} for r in rows
        ]

    def insert_message(self, session_id, role, content, tool_name="none", tool_args=None) -> int:
        with self._lock, self.conn:
            cursor = self.conn.cursor()
            cursor.execute(
                """INSERT INTO messages (session_id, role, content, tool_name, tool_args)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, role, content, tool_name, tool_args),
            )
            return cursor.lastrowid

    def get_sliding_window(self, session_id, limit=10) -> list[tuple[str, str]]:
        """Fetch the last N messages, returned oldest-first."""
        with self._lock:
            rows = self.conn.execute(
                """SELECT role, content FROM messages
                   WHERE session_id = ?
                   ORDER BY message_id DESC
                   LIMIT ?""",
                (session_id, limit),
            ).fetchall()
        return rows[::-1]

    def get_history(self, session_id: str) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                """SELECT role, content, timestamp, tool_name
                   FROM messages WHERE session_id = ?
                   ORDER BY message_id ASC""",
                (session_id,),
            ).fetchall()
        return [
            {"role": r[0], "content": r[1], "timestamp": r[2], "tool_name": r[3]}
            for r in rows
        ]

    def flag_good_message(self, message_id):
        with self._lock, self.conn:
            self.conn.execute(
                "UPDATE messages SET quality_flag = 1 WHERE message_id = ?",
                (message_id,),
            )

    def delete_session(self, session_id):
        with self._lock, self.conn:
            self.conn.execute("DELETE FROM session WHERE session_id = ?", (session_id,))

    # ------------------------------------------------------------------
    #  User profile
    # ------------------------------------------------------------------
    def get_user_profile(self) -> dict[str, str]:
        with self._lock:
            rows = self.conn.execute("SELECT key, value FROM user_profile").fetchall()
        return dict(rows)

    def set_user_profile(self, key: str, value: str):
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO user_profile (key, value, updated_at) "
                "VALUES (?, ?, CURRENT_TIMESTAMP) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
                "updated_at=CURRENT_TIMESTAMP",
                (key, value),
            )

    # ------------------------------------------------------------------
    #  Reminders
    # ------------------------------------------------------------------
    def add_reminder(self, text: str, remind_at: str) -> int:
        with self._lock, self.conn:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO reminders (text, remind_at) VALUES (?, ?)",
                (text, remind_at),
            )
            return cursor.lastrowid

    def get_due_reminders(self, current_iso: str | None = None) -> list[tuple[int, str, str]]:
        """Pending reminders whose time has arrived. Polled by the daemon."""
        if current_iso is None:
            current_iso = datetime.now().isoformat(timespec="seconds")
        with self._lock:
            return self.conn.execute(
                """SELECT reminder_id, text, remind_at FROM reminders
                   WHERE status = 'pending' AND remind_at <= ?
                   ORDER BY remind_at ASC""",
                (current_iso,),
            ).fetchall()

    def get_pending_reminders(self, limit: int = 10) -> list[tuple[int, str, str]]:
        with self._lock:
            return self.conn.execute(
                """SELECT reminder_id, text, remind_at FROM reminders
                   WHERE status = 'pending'
                   ORDER BY remind_at ASC LIMIT ?""",
                (limit,),
            ).fetchall()

    def mark_reminder_completed(self, reminder_id: int):
        with self._lock, self.conn:
            self.conn.execute(
                "UPDATE reminders SET status = 'completed' WHERE reminder_id = ?",
                (reminder_id,),
            )

    # ------------------------------------------------------------------
    #  Maintenance
    # ------------------------------------------------------------------
    def checkpoint(self):
        """Fold the WAL back into the main database file.

        Without this the -wal file grows unbounded; it had reached 2.6MB
        against a 104KB database.
        """
        try:
            with self._lock:
                self.conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.Error as exc:
            log.warning("WAL checkpoint failed: %s", exc)

    def ping(self) -> None:
        with self._lock:
            self.conn.execute("SELECT 1").fetchone()

    def close(self):
        self.checkpoint()
        with self._lock:
            self.conn.close()


# ---------------------------------------------------------------------------
#  Process-wide singleton
# ---------------------------------------------------------------------------
_instance: DatabaseManager | None = None
_instance_lock = threading.Lock()


def get_db() -> DatabaseManager:
    """Return the shared DatabaseManager, creating it on first use."""
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = DatabaseManager()
    return _instance


def close_db() -> None:
    global _instance
    with _instance_lock:
        if _instance is not None:
            _instance.close()
            _instance = None
