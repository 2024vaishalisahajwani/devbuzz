import os
import json
import sqlite3
import pandas as pd

def load_github_metrics(full_name: str) -> dict:
    safe = full_name.replace("/", "_")
    with open(f"data/raw/{safe}.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return {"stars": data.get("stargazers_count", 0), "forks": data.get("forks_count", 0)}

def load_stackoverflow_metrics(full_name: str) -> dict:
    safe = full_name.replace("/", "_")
    with open(f"data/raw/stackoverflow/{safe}.json", "r", encoding="utf-8") as f:
        items = json.load(f)
    return {
        "so_mentions": len(items),
        "so_score_sum": sum(item.get("score", 0) for item in items),
    }

def load_hn_metrics(full_name: str) -> dict:
    safe = full_name.replace("/", "_")
    with open(f"data/raw/hackernews/{safe}.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    items = data["hits"]
    return {
        "hn_mentions": data["total_hits"],   # true total, not len(items)
        "hn_points_sum": sum(item.get("points", 0) or 0 for item in items),
    }
    
def min_max_normalize(series: pd.Series) -> pd.Series:
    """Scale a column to 0-1. Returns all 0s if every value is identical (avoids divide-by-zero)."""
    min_val, max_val = series.min(), series.max()
    if max_val == min_val:
        return series * 0
    return (series - min_val) / (max_val - min_val)

def build_buzz_table() -> pd.DataFrame:
    conn = sqlite3.connect("db/devbuzz.db")
    repos = pd.read_sql("SELECT repo_id, full_name FROM repos", conn)
    conn.close()

    rows = []
    for _, repo in repos.iterrows():
        metrics = {"repo_id": repo["repo_id"], "full_name": repo["full_name"]}
        metrics.update(load_github_metrics(repo["full_name"]))
        metrics.update(load_stackoverflow_metrics(repo["full_name"]))
        metrics.update(load_hn_metrics(repo["full_name"]))
        rows.append(metrics)

    df = pd.DataFrame(rows)

    for col in ["stars", "so_mentions", "so_score_sum", "hn_mentions", "hn_points_sum"]:
        df[f"norm_{col}"] = min_max_normalize(df[col])

    df["so_activity"] = (df["norm_so_mentions"] + df["norm_so_score_sum"]) / 2
    df["hn_activity"] = (df["norm_hn_mentions"] + df["norm_hn_points_sum"]) / 2

    df["buzz_score"] = (
        0.5 * df["norm_stars"] + 0.25 * df["so_activity"] + 0.25 * df["hn_activity"]
    ).round(4)

    return df.sort_values("buzz_score", ascending=False)

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    df = build_buzz_table()
    df.to_csv("data/processed/buzz_scores.csv", index=False)
    print(df[["full_name", "stars", "so_mentions", "hn_mentions", "buzz_score"]].to_string(index=False))