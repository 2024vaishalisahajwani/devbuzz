import sqlite3
import pandas as pd
from datetime import date

def load_buzz_daily():
    df = pd.read_csv("data/processed/buzz_scores.csv")
    today = date.today().isoformat()  # e.g. "2026-09-18"   

    conn = sqlite3.connect("db/devbuzz.db")
    cursor = conn.cursor()

    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO buzz_daily
                (repo_id, date, stars, forks, open_issues, so_mentions, hn_mentions, hn_points, buzz_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(repo_id, date) DO UPDATE SET
                stars=excluded.stars,
                hn_mentions=excluded.hn_mentions,
                hn_points=excluded.hn_points,
                buzz_score=excluded.buzz_score
        """, (
            int(row["repo_id"]), today, int(row["stars"]), None, None,
            int(row["so_mentions"]),
            int(row["hn_mentions"]), int(row["hn_points_sum"]), float(row["buzz_score"])
        ))

    conn.commit()

    cursor.execute("SELECT COUNT(*) FROM buzz_daily WHERE date = ?", (today,))
    count = cursor.fetchone()[0]
    print(f"buzz_daily now has {count} rows for {today}.")

    conn.close()

if __name__ == "__main__":
    load_buzz_daily()