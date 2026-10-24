import sqlite3
import os

def create_tables():
    os.makedirs("db", exist_ok=True)
    conn = sqlite3.connect("db/devbuzz.db")
    cursor = conn.cursor()

    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS repos (
        repo_id INTEGER PRIMARY KEY AUTOINCREMENT,
        full_name TEXT UNIQUE NOT NULL,
        owner TEXT NOT NULL,
        name TEXT NOT NULL,
        language TEXT,
        github_created_at TEXT
    );

    CREATE TABLE IF NOT EXISTS buzz_daily (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        repo_id INTEGER NOT NULL,
        date TEXT NOT NULL,
        stars INTEGER,
        forks INTEGER,
        open_issues INTEGER,
        so_mentions INTEGER,
        hn_mentions INTEGER,
        hn_points INTEGER,
        buzz_score REAL,
        FOREIGN KEY (repo_id) REFERENCES repos(repo_id),
        UNIQUE(repo_id, date)
    );
    """)

    conn.commit()
    conn.close()
    print("Tables created (or already existed).")

if __name__ == "__main__":
    create_tables()