import sqlite3
import uuid

class DatabaseManager:
    def __init__(self, db_path='gideon.db'):
        # 1. Establish connection and bind to instance
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        
        # 2. Apply performance & relational optimizations
        self.conn.execute('PRAGMA journal_mode=WAL')
        self.conn.execute('PRAGMA synchronous=NORMAL')
        self.conn.execute('PRAGMA foreign_keys=ON')
        
        # 3. Initialize schema
        with self.conn:
            cursor = self.conn.cursor()
            
            # Session Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session (
                    session_id TEXT PRIMARY KEY,
                    session_name TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Messages Table (Updated with tool tracking and cascade deletion)
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
            
            # Indexing for fast sliding window retrieval
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_messages_session_id 
                ON messages(session_id)
            """)

    # ==========================================
    # CRUD HELPER FUNCTIONS
    # ==========================================

    def create_session(self, session_name="New Conversation"):
        """Creates a new chat session and returns the unique string ID."""
        session_id = str(uuid.uuid4())
        
        with self.conn:
            self.conn.execute(
                "INSERT INTO session (session_id, session_name) VALUES (?, ?)", 
                (session_id, session_name)
            )
        return session_id

    def insert_message(self, session_id, role, content, tool_name='none', tool_args=None):
        """Saves a message and returns the numeric primary key for ChromaDB."""
        with self.conn:
            cursor = self.conn.cursor()
            cursor.execute(
                """INSERT INTO messages (session_id, role, content, tool_name, tool_args) 
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, role, content, tool_name, tool_args)
            )
            return cursor.lastrowid

    def get_sliding_window(self, session_id, limit=10):
        """Fetches the last N messages and returns them chronologically."""
        cursor = self.conn.cursor()
        cursor.execute(
            """SELECT role, content FROM messages 
               WHERE session_id = ? 
               ORDER BY message_id DESC 
               LIMIT ?""",
            (session_id, limit)
        )
        
        rows = cursor.fetchall()
        # Reverse the list so the oldest is first, newest is last
        return rows[::-1]

    def flag_good_message(self, message_id):
        """Flags an assistant's message as high-quality for future fine-tuning."""
        with self.conn:
            self.conn.execute(
                "UPDATE messages SET quality_flag = 1 WHERE message_id = ?", 
                (message_id,)
            )

    def delete_session(self, session_id):
        """Deletes a session. ON DELETE CASCADE automatically removes its messages."""
        with self.conn:
            self.conn.execute(
                "DELETE FROM session WHERE session_id = ?", 
                (session_id,)
            )

    def close(self):
        """Gracefully closes the database connection."""
        self.conn.close()