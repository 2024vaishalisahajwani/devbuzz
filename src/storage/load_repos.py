import sqlite3
import json
import glob

def load_repos():
    conn = sqlite3.connect("db/devbuzz.db")
    cursor = conn.cursor()

    files = glob.glob("data/raw/*.json")
    for filepath in files:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        full_name = data["full_name"]          # e.g. "astral-sh/uv"
        owner, name = full_name.split("/")

        cursor.execute("""
            INSERT OR IGNORE INTO repos (full_name, owner, name, language, github_created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (full_name, owner, name, data["language"], data["created_at"]))

    conn.commit()

    # quick check
    cursor.execute("SELECT COUNT(*) FROM repos")
    count = cursor.fetchone()[0]
    print(f"repos table now has {count} rows.")

    conn.close()

if __name__ == "__main__":
    load_repos()