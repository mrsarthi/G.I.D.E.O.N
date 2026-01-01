import sqlite3
from datetime import datetime

def log_command(text, category="Pending"):
    conn = sqlite3.connect('GideonLog.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            raw_text TEXT NOT NULL,
            timestamp TEXT,
            category TEXT
        )
    ''')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute("INSERT INTO history (raw_text, timestamp, category) VALUES (?, ?, ?)", 
                   (text, timestamp, category))
    conn.commit()
    conn.close()