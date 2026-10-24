import sqlite3
import os
import pandas as pd
import matplotlib.pyplot as plt

def load_buzz_daily() -> pd.DataFrame:
    conn = sqlite3.connect("db/devbuzz.db")
    query = """
        SELECT b.*, r.full_name
        FROM buzz_daily b
        JOIN repos r ON b.repo_id = r.repo_id
        ORDER BY b.date, b.buzz_score DESC
    """
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def plot_buzz_ranking(df: pd.DataFrame, out_dir: str):
    latest_date = df["date"].max()
    latest = df[df["date"] == latest_date].sort_values("buzz_score", ascending=True)

    plt.figure(figsize=(8, 6))
    plt.barh(latest["full_name"], latest["buzz_score"], color="#4C72B0")
    plt.xlabel("Buzz Score")
    plt.title(f"Repo Buzz Score Ranking ({latest_date})")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/buzz_ranking.png", dpi=150)
    plt.close()

def plot_stars_vs_buzz(df: pd.DataFrame, out_dir: str):
    latest_date = df["date"].max()
    latest = df[df["date"] == latest_date]

    plt.figure(figsize=(8, 6))
    plt.scatter(latest["stars"], latest["buzz_score"], color="#DD8452")
    for _, row in latest.iterrows():
        plt.annotate(row["full_name"].split("/")[-1], (row["stars"], row["buzz_score"]),
                     fontsize=8, xytext=(4, 4), textcoords="offset points")
    plt.xscale("log")
    plt.xlabel("GitHub Stars (log scale)")
    plt.ylabel("Buzz Score")
    plt.title(f"Stars vs. Buzz Score ({latest_date})")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/stars_vs_buzz.png", dpi=150)
    plt.close()

def plot_buzz_trend(df: pd.DataFrame, out_dir: str):
    plt.figure(figsize=(9, 6))
    for full_name, group in df.groupby("full_name"):
        group = group.sort_values("date")
        plt.plot(group["date"], group["buzz_score"], marker="o", label=full_name.split("/")[-1])
    plt.xlabel("Date")
    plt.ylabel("Buzz Score")
    plt.title("Buzz Score Trend Over Time")
    plt.legend(fontsize=7, ncol=2)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{out_dir}/buzz_trend.png", dpi=150)
    plt.close()

def compute_correlation(df: pd.DataFrame) -> dict:
    latest_date = df["date"].max()
    latest = df[df["date"] == latest_date]
    return {
        "stars_vs_buzz_score": latest["stars"].corr(latest["buzz_score"]),
        # buzz_score is 50% derived from stars by formula design, so the above
        # correlation is partly mechanical/circular. The more meaningful test is
        # stars vs. the RAW community-signal columns, independent of the formula:
        "stars_vs_hn_mentions": latest["stars"].corr(latest["hn_mentions"]),
        "stars_vs_hn_points": latest["stars"].corr(latest["hn_points"]),
    }

if __name__ == "__main__":
    out_dir = "reports/figures"
    os.makedirs(out_dir, exist_ok=True)

    df = load_buzz_daily()

    plot_buzz_ranking(df, out_dir)
    plot_stars_vs_buzz(df, out_dir)
    plot_buzz_trend(df, out_dir)

    corr = compute_correlation(df)
    print(f"Charts saved to {out_dir}/")
    print(f"Correlation stars vs. buzz_score (partly circular, buzz_score includes stars):  {corr['stars_vs_buzz_score']:.3f}")
    print(f"Correlation stars vs. HN mentions (independent signal):                          {corr['stars_vs_hn_mentions']:.3f}")
    print(f"Correlation stars vs. HN points   (independent signal):                          {corr['stars_vs_hn_points']:.3f}")
    