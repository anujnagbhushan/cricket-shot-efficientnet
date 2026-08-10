import sqlite3

DB_NAME = "analysis.db"


def get_connection():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS analysis (

        id INTEGER PRIMARY KEY AUTOINCREMENT,

        session_id TEXT,
        batsman_name TEXT,
        match_format TEXT,
        playing_hand TEXT,

        prediction TEXT,
        confidence REAL,
        inference_time REAL,

        coach_correct TEXT,
        correct_label TEXT,

        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """)

    conn.commit()
    conn.close()