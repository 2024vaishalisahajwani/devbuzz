import os
import sqlite3
import numpy as np
import pandas as pd

MIN_DAYS_FOR_TREND = 3  # minimum days needed for a meaningful trend/target

def load_all_buzz_daily() -> pd.DataFrame:
    conn = sqlite3.connect("db/devbuzz.db")
    query = """
        SELECT b.*, r.full_name
        FROM buzz_daily b
        JOIN repos r ON b.repo_id = r.repo_id
        ORDER BY r.full_name, b.date
    """
    df = pd.read_sql(query, conn)
    conn.close()
    df["date"] = pd.to_datetime(df["date"])
    return df

def compute_features_for_repo(group: pd.DataFrame) -> dict:
    group = group.sort_values("date").reset_index(drop=True)
    n_days = len(group)

    features = {
        "repo_id": group["repo_id"].iloc[0],
        "full_name": group["full_name"].iloc[0],
        "n_days_tracked": n_days,
        "latest_stars": group["stars"].iloc[-1],
        "latest_buzz_score": group["buzz_score"].iloc[-1],
        "avg_buzz_score": group["buzz_score"].mean(),
        "avg_hn_mentions": group["hn_mentions"].mean(),
    }

    if n_days >= 2:
        stars_start, stars_end = group["stars"].iloc[0], group["stars"].iloc[-1]
        features["star_growth_rate"] = (
            (stars_end - stars_start) / stars_start if stars_start > 0 else np.nan
        )
        features["buzz_score_delta"] = group["buzz_score"].iloc[-1] - group["buzz_score"].iloc[0]
    else:
        features["star_growth_rate"] = np.nan
        features["buzz_score_delta"] = np.nan

    if n_days >= MIN_DAYS_FOR_TREND:
        day_index = np.arange(n_days)
        slope, _ = np.polyfit(day_index, group["buzz_score"], 1)
        features["buzz_trend_slope"] = slope
        features["still_trending"] = int(slope > 0)   # TARGET LABEL
    else:
        features["buzz_trend_slope"] = np.nan
        features["still_trending"] = None               # not yet defined — insufficient history

    return features

def build_feature_table() -> pd.DataFrame:
    df = load_all_buzz_daily()
    rows = [compute_features_for_repo(g) for _, g in df.groupby("repo_id")]
    return pd.DataFrame(rows)

if __name__ == "__main__":
    os.makedirs("data/processed", exist_ok=True)
    features_df = build_feature_table()
    features_df.to_csv("data/processed/model_features.csv", index=False)

    n_days_available = int(features_df["n_days_tracked"].max())
    print(f"Feature table built for {len(features_df)} repos. Max days tracked so far: {n_days_available}\n")
    print(features_df[["full_name", "n_days_tracked", "latest_buzz_score", "buzz_trend_slope", "still_trending"]].to_string(index=False))

    if n_days_available < MIN_DAYS_FOR_TREND:
        print(f"\n[NOTICE] Only {n_days_available} day(s) tracked — need at least {MIN_DAYS_FOR_TREND} for a meaningful trend/target label.")
        print("Keep running `python src\\run_pipeline.py` daily, then re-run this script before Milestone 10.")