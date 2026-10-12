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

import json
import sqlite3
import threading
import uuid
from dataclasses import fields
from datetime import datetime
from pathlib import Path

from gideon.core.config import DB_PATH
from gideon.core.logs import get_logger
from gideon.storage.models import ChatMessage, Note, PendingAction, Project, Reminder

log = get_logger("gideon.db")

_SCHEMA_ROLES = ("user", "assistant", "tool", "tool_call", "system")


def _inserted_id(cursor: sqlite3.Cursor) -> int:
    """Row id of the INSERT just executed (never None after a successful one)."""
    assert cursor.lastrowid is not None
    return cursor.lastrowid


_REMINDER_COLUMNS = "reminder_id, text, remind_at, channel, chat_id, remote_synced"

_PROJECT_FIELDS = tuple(f.name for f in fields(Project))
_PROJECT_COLUMNS = ", ".join(_PROJECT_FIELDS)


def _project_value(project: Project, field: str) -> object:
    value = getattr(project, field)
    return int(value) if isinstance(value, bool) else value


def _project(row: tuple) -> Project:
    project = Project(**dict(zip(_PROJECT_FIELDS, row, strict=True)))
    project.missing = bool(project.missing)
    return project


def _reminder(row: tuple) -> Reminder:
    """Build a Reminder from a row selected as _REMINDER_COLUMNS."""
    reminder_id, text, remind_at, channel, chat_id, remote_synced = row
    return Reminder(
        reminder_id=reminder_id,
        text=text,
        remind_at=remind_at,
        # Rows predating channel-aware delivery may hold NULL here.
        channel=channel or "web",
        chat_id=chat_id,
        remote_synced=bool(remote_synced),
    )


_ACTION_COLUMNS = (
    "action_id, session_id, channel, tool_name, tool_args, status, result, "
    "created_at, resolved_at, trace_id"
)


def _action(row: tuple) -> PendingAction:
    (action_id, session_id, channel, tool_name, tool_args,
     status, result, created_at, resolved_at, trace_id) = row
    try:
        args = json.loads(tool_args) if tool_args else {}
    except (TypeError, ValueError):
        args = {}
    return PendingAction(
        action_id=action_id,
        session_id=session_id,
        channel=channel,
        tool_name=tool_name,
        tool_args=args if isinstance(args, dict) else {},
        status=status,
        created_at=created_at,
        result=result,
        resolved_at=resolved_at,
        trace_id=trace_id,
    )


_TRACE_COLUMNS = (
    "trace_id, session_id, channel, created_at, request, model, placement, "
    "tools_offered, outcome, reply, error, total_ms, first_text_ms, detail, "
    "rating, correction_kind, correction_text, corrected_at"
)


def _trace(row: tuple) -> dict:
    record = dict(zip(_TRACE_COLUMNS.split(", "), row, strict=True))
    try:
        record["detail"] = json.loads(record["detail"] or "{}")
    except (TypeError, ValueError):
        record["detail"] = {}
    record["tools_offered"] = bool(record["tools_offered"])
    return record


class DatabaseManager:
    def __init__(self, db_path=None):
        self.db_path = str(db_path or DB_PATH)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
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
                    -- Where the reminder was created. chat_id lets the daemon
                    -- reply into the same Telegram conversation rather than
                    -- broadcasting to every allowed user.
                    channel TEXT DEFAULT 'web',
                    chat_id TEXT,
                    -- 1 once the always-on scheduler has accepted this row.
                    -- Delivery ownership follows this flag: synced rows are
                    -- pushed to Telegram by the remote node (which fires even
                    -- when this machine is off), unsynced rows stay this
                    -- process's responsibility. Exactly one side sends, so
                    -- there is nothing to de-duplicate.
                    remote_synced INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_reminders_status_time
                ON reminders(status, remind_at)
            """)

            # Durable record of every shell command attempted, including the
            # ones that were refused. Log files and the event bus are both
            # ephemeral; for a feature that runs arbitrary code you want to be
            # able to answer "what did it run, when, from where" afterwards.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS command_audit (
                    audit_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    command TEXT NOT NULL,
                    cwd TEXT,
                    channel TEXT DEFAULT 'unknown',
                    allowed INTEGER NOT NULL,
                    block_reason TEXT,
                    return_code INTEGER,
                    output_excerpt TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_command_audit_time
                ON command_audit(created_at DESC)
            """)

            # Privileged tool calls awaiting the user's approval, and the
            # outcome once they have it. Timestamps are local ISO strings
            # written by Python: CURRENT_TIMESTAMP is UTC, and expiry
            # arithmetic against local time would be off by the UTC offset.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pending_actions (
                    action_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    tool_name TEXT NOT NULL,
                    tool_args TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    result TEXT,
                    created_at TEXT NOT NULL,
                    resolved_at TEXT
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_pending_actions_session
                ON pending_actions(session_id, action_id DESC)
            """)

            # One structured record per agent turn (PRD §6.3): what was asked,
            # what the harness and model decided, what happened, how long each
            # stage took, and — added later — the user's correction. The eval
            # suite and any future training data are built from these rows.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    channel TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    request TEXT NOT NULL,
                    model TEXT,
                    placement TEXT,
                    tools_offered INTEGER,
                    outcome TEXT,
                    reply TEXT,
                    error TEXT,
                    total_ms INTEGER,
                    first_text_ms INTEGER,
                    detail TEXT NOT NULL,
                    rating INTEGER,
                    correction_kind TEXT,
                    correction_text TEXT,
                    corrected_at TEXT
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_traces_session
                ON traces(session_id, created_at DESC)
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS notes (
                    note_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    kind TEXT NOT NULL DEFAULT 'note',
                    text TEXT NOT NULL,
                    done INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                )
            """)

            # Fact memory, first kind of fact: the user's code projects and
            # their state when last looked at (storage/projects.py). Rebuilt
            # from disk on every refresh except `opened_with`, which records
            # the user's own choice of app.
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    path TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    kind TEXT,
                    description TEXT,
                    branch TEXT,
                    dirty INTEGER,
                    last_commit_at TEXT,
                    last_commit TEXT,
                    modified_at TEXT,
                    last_opened_at TEXT,
                    opened_with TEXT,
                    summary TEXT NOT NULL,
                    refreshed_at TEXT NOT NULL,
                    missing INTEGER NOT NULL DEFAULT 0
                )
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

            # Reminders predating channel-aware delivery have no origin.
            existing = {r[1] for r in cursor.execute("PRAGMA table_info(reminders)")}
            for column, ddl in (
                ("channel", "ALTER TABLE reminders ADD COLUMN channel TEXT DEFAULT 'web'"),
                ("chat_id", "ALTER TABLE reminders ADD COLUMN chat_id TEXT"),
                # Rows predating the remote scheduler default to unsynced, so
                # this process keeps delivering them until the mirror picks
                # them up. Defaulting the other way would silently drop them.
                (
                    "remote_synced",
                    "ALTER TABLE reminders ADD COLUMN remote_synced INTEGER DEFAULT 0",
                ),
            ):
                if column not in existing:
                    cursor.execute(ddl)
                    log.info("Added reminders.%s column", column)

            # 1 while a session still carries a placeholder name that should
            # be replaced by a generated title after its first exchange.
            # Links an approval request to the turn that raised it, so a denial
            # is recorded as a correction on that turn's trace.
            existing = {r[1] for r in cursor.execute("PRAGMA table_info(pending_actions)")}
            if "trace_id" not in existing:
                cursor.execute("ALTER TABLE pending_actions ADD COLUMN trace_id TEXT")
                log.info("Added pending_actions.trace_id column")

            existing = {r[1] for r in cursor.execute("PRAGMA table_info(session)")}
            if "auto_title" not in existing:
                cursor.execute("ALTER TABLE session ADD COLUMN auto_title INTEGER DEFAULT 0")
                log.info("Added session.auto_title column")

    # ------------------------------------------------------------------
    #  Sessions & messages
    # ------------------------------------------------------------------
    def create_session(self, session_name="New Conversation", auto_title: bool = False) -> str:
        """Create a session. auto_title marks the name as a placeholder to be
        replaced by a generated title once the conversation has started."""
        session_id = str(uuid.uuid4())
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO session (session_id, session_name, auto_title) VALUES (?, ?, ?)",
                (session_id, session_name, 1 if auto_title else 0),
            )
        return session_id

    def session_wants_title(self, session_id: str) -> bool:
        with self._lock:
            row = self.conn.execute(
                "SELECT auto_title FROM session WHERE session_id = ?", (session_id,)
            ).fetchone()
        return bool(row and row[0])

    def rename_session(self, session_id: str, session_name: str) -> None:
        """Set a session's name and stop treating it as a placeholder."""
        with self._lock, self.conn:
            self.conn.execute(
                "UPDATE session SET session_name = ?, auto_title = 0 WHERE session_id = ?",
                (session_name, session_id),
            )

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
            return _inserted_id(cursor)

    def get_sliding_window(self, session_id, limit=10) -> list[ChatMessage]:
        """Fetch the last N messages, returned oldest-first."""
        with self._lock:
            rows = self.conn.execute(
                """SELECT role, content FROM messages
                   WHERE session_id = ?
                   ORDER BY message_id DESC
                   LIMIT ?""",
                (session_id, limit),
            ).fetchall()
        return [ChatMessage(role, content) for role, content in reversed(rows)]

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
    def add_reminder(
        self,
        text: str,
        remind_at: str,
        channel: str = "web",
        chat_id: str | None = None,
    ) -> int:
        with self._lock, self.conn:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO reminders (text, remind_at, channel, chat_id) "
                "VALUES (?, ?, ?, ?)",
                (text, remind_at, channel, chat_id),
            )
            return _inserted_id(cursor)

    def get_due_reminders(self, current_iso: str | None = None) -> list[Reminder]:
        """Pending reminders whose time has arrived. Polled by the daemon."""
        if current_iso is None:
            current_iso = datetime.now().isoformat(timespec="seconds")
        with self._lock:
            rows = self.conn.execute(
                f"""SELECT {_REMINDER_COLUMNS}
                   FROM reminders
                   WHERE status = 'pending' AND remind_at <= ?
                   ORDER BY remind_at ASC""",
                (current_iso,),
            ).fetchall()
        return [_reminder(r) for r in rows]

    def get_pending_reminders(self, limit: int = 10) -> list[Reminder]:
        with self._lock:
            rows = self.conn.execute(
                f"""SELECT {_REMINDER_COLUMNS}
                   FROM reminders
                   WHERE status = 'pending'
                   ORDER BY remind_at ASC LIMIT ?""",
                (limit,),
            ).fetchall()
        return [_reminder(r) for r in rows]

    def mark_reminder_completed(self, reminder_id: int):
        with self._lock, self.conn:
            self.conn.execute(
                "UPDATE reminders SET status = 'completed' WHERE reminder_id = ?",
                (reminder_id,),
            )

    def mark_reminder_missed(self, reminder_id: int):
        """Retire a reminder that came due while nothing was running.

        Kept distinct from 'completed' so the digest can say what was missed
        and so the audit trail does not claim it was delivered on time.
        """
        with self._lock, self.conn:
            self.conn.execute(
                "UPDATE reminders SET status = 'missed' WHERE reminder_id = ?",
                (reminder_id,),
            )

    # ------------------------------------------------------------------
    #  Remote scheduler mirror
    # ------------------------------------------------------------------
    def get_unsynced_reminders(self, limit: int = 50) -> list[Reminder]:
        """Pending future reminders the always-on node has not accepted yet.

        Past-due rows are excluded: mirroring them would ask the node to fire
        something that is already late, which the local digest handles better.
        """
        now_iso = datetime.now().isoformat(timespec="seconds")
        with self._lock:
            rows = self.conn.execute(
                f"""SELECT {_REMINDER_COLUMNS}
                   FROM reminders
                   WHERE status = 'pending' AND remote_synced = 0 AND remind_at > ?
                   ORDER BY remind_at ASC LIMIT ?""",
                (now_iso, limit),
            ).fetchall()
        return [_reminder(r) for r in rows]

    def mark_reminder_synced(self, reminder_id: int, synced: bool = True):
        with self._lock, self.conn:
            self.conn.execute(
                "UPDATE reminders SET remote_synced = ? WHERE reminder_id = ?",
                (1 if synced else 0, reminder_id),
            )

    # ------------------------------------------------------------------
    #  Pending actions (approval queue)
    # ------------------------------------------------------------------
    def create_action(
        self,
        session_id: str,
        channel: str,
        tool_name: str,
        tool_args: dict,
        trace_id: str | None = None,
    ) -> PendingAction:
        created = datetime.now().isoformat(timespec="seconds")
        with self._lock, self.conn:
            cursor = self.conn.cursor()
            cursor.execute(
                """INSERT INTO pending_actions
                   (session_id, channel, tool_name, tool_args, created_at, trace_id)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (session_id, channel, tool_name, json.dumps(tool_args), created, trace_id),
            )
            action_id = _inserted_id(cursor)
        return PendingAction(
            action_id=action_id,
            session_id=session_id,
            channel=channel,
            tool_name=tool_name,
            tool_args=dict(tool_args),
            status="pending",
            created_at=created,
            trace_id=trace_id,
        )

    def get_action(self, action_id: int) -> PendingAction | None:
        with self._lock:
            row = self.conn.execute(
                f"SELECT {_ACTION_COLUMNS} FROM pending_actions WHERE action_id = ?",
                (action_id,),
            ).fetchone()
        return _action(row) if row else None

    def resolve_action(
        self, action_id: int, status: str, result: str | None = None, expect: str = "pending"
    ) -> bool:
        """Move an action out of `expect` status. False if it was not in it —
        which is what stops one approval executing a command twice."""
        with self._lock, self.conn:
            cursor = self.conn.execute(
                """UPDATE pending_actions SET status = ?, result = ?, resolved_at = ?
                   WHERE action_id = ? AND status = ?""",
                (status, result, datetime.now().isoformat(timespec="seconds"), action_id, expect),
            )
            return cursor.rowcount == 1

    def list_actions(
        self,
        session_id: str | None = None,
        status: str | None = None,
        since: str | None = None,
        limit: int = 20,
    ) -> list[PendingAction]:
        clauses, params = [], []
        if session_id is not None:
            clauses.append("session_id = ?")
            params.append(session_id)
        if status is not None:
            clauses.append("status = ?")
            params.append(status)
        if since is not None:
            clauses.append("created_at >= ?")
            params.append(since)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self.conn.execute(
                f"SELECT {_ACTION_COLUMNS} FROM pending_actions {where} "
                "ORDER BY action_id DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [_action(r) for r in rows]

    def expire_actions(self, created_before: str) -> int:
        with self._lock, self.conn:
            cursor = self.conn.execute(
                """UPDATE pending_actions SET status = 'expired', resolved_at = ?
                   WHERE status = 'pending' AND created_at < ?""",
                (datetime.now().isoformat(timespec="seconds"), created_before),
            )
            return cursor.rowcount

    # ------------------------------------------------------------------
    #  Traces
    # ------------------------------------------------------------------
    def insert_trace(self, record: dict) -> None:
        """Store one turn's trace. `detail` holds the nested per-pass data."""
        with self._lock, self.conn:
            self.conn.execute(
                """INSERT OR REPLACE INTO traces
                   (trace_id, session_id, channel, created_at, request, model, placement,
                    tools_offered, outcome, reply, error, total_ms, first_text_ms, detail)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    record["trace_id"],
                    record["session_id"],
                    record["channel"],
                    record["created_at"],
                    record["request"],
                    record.get("model"),
                    record.get("placement"),
                    1 if record.get("tools_offered") else 0,
                    record.get("outcome"),
                    record.get("reply"),
                    record.get("error"),
                    record.get("total_ms"),
                    record.get("first_text_ms"),
                    json.dumps(record.get("detail") or {}),
                ),
            )

    def annotate_trace(self, trace_id: str, fields: dict) -> None:
        """Merge extra measurements into a trace's detail JSON."""
        with self._lock, self.conn:
            row = self.conn.execute(
                "SELECT detail FROM traces WHERE trace_id = ?", (trace_id,)
            ).fetchone()
            if row is None:
                return
            try:
                detail = json.loads(row[0] or "{}")
            except (TypeError, ValueError):
                detail = {}
            detail.update(fields)
            self.conn.execute(
                "UPDATE traces SET detail = ? WHERE trace_id = ?", (json.dumps(detail), trace_id)
            )

    def get_trace(self, trace_id: str) -> dict | None:
        with self._lock:
            row = self.conn.execute(
                f"SELECT {_TRACE_COLUMNS} FROM traces WHERE trace_id = ?", (trace_id,)
            ).fetchone()
        return _trace(row) if row else None

    def latest_trace(self, session_id: str) -> dict | None:
        with self._lock:
            row = self.conn.execute(
                f"SELECT {_TRACE_COLUMNS} FROM traces WHERE session_id = ? "
                "ORDER BY created_at DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        return _trace(row) if row else None

    def list_traces(self, limit: int = 50, since: str | None = None) -> list[dict]:
        where, params = ("WHERE created_at >= ?", [since]) if since else ("", [])
        with self._lock:
            rows = self.conn.execute(
                f"SELECT {_TRACE_COLUMNS} FROM traces {where} "
                "ORDER BY created_at DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [_trace(r) for r in rows]

    def record_correction(
        self,
        trace_id: str,
        kind: str | None = None,
        text: str | None = None,
        rating: int | None = None,
    ) -> bool:
        """Attach the user's correction to a turn. Ratings overwrite; a
        correction never erases an earlier rating, and vice versa."""
        with self._lock, self.conn:
            cursor = self.conn.execute(
                """UPDATE traces SET
                       correction_kind = COALESCE(?, correction_kind),
                       correction_text = COALESCE(?, correction_text),
                       rating = COALESCE(?, rating),
                       corrected_at = ?
                   WHERE trace_id = ?""",
                (kind, text, rating, datetime.now().isoformat(timespec="seconds"), trace_id),
            )
            return cursor.rowcount == 1

    # ------------------------------------------------------------------
    #  Notes & tasks
    # ------------------------------------------------------------------
    def add_note(self, text: str, kind: str = "note") -> int:
        with self._lock, self.conn:
            cursor = self.conn.cursor()
            cursor.execute(
                "INSERT INTO notes (kind, text, created_at) VALUES (?, ?, ?)",
                (kind, text, datetime.now().isoformat(timespec="seconds")),
            )
            return _inserted_id(cursor)

    def list_notes(
        self,
        kind: str | None = None,
        query: str | None = None,
        include_done: bool = False,
        limit: int = 20,
    ) -> list[Note]:
        clauses, params = [], []
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        if query:
            clauses.append("text LIKE ?")
            params.append(f"%{query}%")
        if not include_done:
            clauses.append("done = 0")
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        with self._lock:
            rows = self.conn.execute(
                f"SELECT note_id, kind, text, done, created_at FROM notes {where} "
                "ORDER BY note_id DESC LIMIT ?",
                (*params, limit),
            ).fetchall()
        return [Note(r[0], r[1], r[2], bool(r[3]), r[4]) for r in rows]

    def complete_note(self, note_id: int) -> bool:
        with self._lock, self.conn:
            cursor = self.conn.execute(
                "UPDATE notes SET done = 1 WHERE note_id = ? AND done = 0", (note_id,)
            )
            return cursor.rowcount == 1

    # ------------------------------------------------------------------
    #  Projects (fact memory)
    # ------------------------------------------------------------------
    def upsert_project(self, project: Project) -> None:
        """Store a freshly scanned project; keeps the user's `opened_with`."""
        with self._lock, self.conn:
            self.conn.execute(
                f"INSERT INTO projects ({_PROJECT_COLUMNS}) "
                f"VALUES ({', '.join('?' * len(_PROJECT_FIELDS))}) "
                "ON CONFLICT(path) DO UPDATE SET "
                + ", ".join(
                    f"{f} = excluded.{f}" for f in _PROJECT_FIELDS if f not in ("path", "opened_with")
                ),
                tuple(_project_value(project, f) for f in _PROJECT_FIELDS),
            )

    def list_projects(self, include_missing: bool = False) -> list[Project]:
        where = "" if include_missing else "WHERE missing = 0"
        with self._lock:
            rows = self.conn.execute(
                f"SELECT {_PROJECT_COLUMNS} FROM projects {where} ORDER BY name COLLATE NOCASE"
            ).fetchall()
        return [_project(r) for r in rows]

    def mark_projects_missing(self, keep: set[str]) -> list[str]:
        """Flag every project not in `keep` as gone; returns the newly missing paths."""
        with self._lock, self.conn:
            gone = [
                r[0]
                for r in self.conn.execute("SELECT path FROM projects WHERE missing = 0")
                if r[0] not in keep
            ]
            self.conn.executemany("UPDATE projects SET missing = 1 WHERE path = ?", [(p,) for p in gone])
        return gone

    def set_project_opener(self, path: str, app: str) -> None:
        with self._lock, self.conn:
            self.conn.execute("UPDATE projects SET opened_with = ? WHERE path = ?", (app, path))

    # ------------------------------------------------------------------
    #  Command audit
    # ------------------------------------------------------------------
    def record_command(
        self,
        command: str,
        cwd: str | None,
        channel: str,
        allowed: bool,
        block_reason: str | None = None,
        return_code: int | None = None,
        output_excerpt: str | None = None,
    ) -> int:
        """Append one shell-command attempt to the audit trail."""
        with self._lock, self.conn:
            cursor = self.conn.cursor()
            cursor.execute(
                """INSERT INTO command_audit
                   (command, cwd, channel, allowed, block_reason, return_code, output_excerpt)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    command,
                    cwd,
                    channel,
                    1 if allowed else 0,
                    block_reason,
                    return_code,
                    (output_excerpt or "")[:500] or None,
                ),
            )
            return _inserted_id(cursor)

    def get_command_audit(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self.conn.execute(
                """SELECT audit_id, command, cwd, channel, allowed, block_reason,
                          return_code, output_excerpt, created_at
                   FROM command_audit ORDER BY audit_id DESC LIMIT ?""",
                (limit,),
            ).fetchall()
        keys = (
            "audit_id", "command", "cwd", "channel", "allowed", "block_reason",
            "return_code", "output_excerpt", "created_at",
        )
        return [dict(zip(keys, r, strict=True)) for r in rows]

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
