import sqlite3, json
from datetime import datetime

DB = "profiles.db"

def init_db():
    con = sqlite3.connect(DB)
    con.execute("""
        CREATE TABLE IF NOT EXISTS profiles (
            user_id TEXT PRIMARY KEY,
            data TEXT NOT NULL,
            updated_at TEXT
        )
    """)
    con.commit()
    con.close()

def load_profile(user_id: str) -> dict:
    con = sqlite3.connect(DB)
    row = con.execute("SELECT data FROM profiles WHERE user_id=?", (user_id,)).fetchone()
    con.close()
    return json.loads(row[0]) if row else {}

def save_profile(user_id: str, profile: dict):
    con = sqlite3.connect(DB)
    con.execute("""
        INSERT INTO profiles (user_id, data, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(user_id) DO UPDATE SET data=excluded.data, updated_at=excluded.updated_at
    """, (user_id, json.dumps(profile), datetime.now().isoformat()))
    con.commit()
    con.close()
