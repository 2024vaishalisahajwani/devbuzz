import os
import sys
import time
import subprocess
import sqlite3
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

DB_PATH = "db/devbuzz.db"

st.set_page_config(page_title="DevBuzz Dashboard", layout="wide")

# --- Data loading (cached so the DB isn't re-queried on every click) ---
@st.cache_data
def load_buzz_daily():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("""
        SELECT b.*, r.full_name, r.owner, r.name
        FROM buzz_daily b JOIN repos r ON b.repo_id = r.repo_id
        ORDER BY b.date
    """, conn)
    conn.close()
    return df

@st.cache_data
def load_features():
    return pd.read_csv("data/processed/model_features.csv")

df = load_buzz_daily()
latest = df.sort_values("date").groupby("full_name").tail(1)

# --- Header ---
st.title("DevBuzz — Live Dashboard")
st.caption("Open-Source Trend & Community Engagement Pipeline")

# The trend chart shows every repo; use the chart legend to hide or isolate lines.
selected_repos = sorted(df["full_name"].unique())

# --- Top metrics ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Repos tracked", df["full_name"].nunique())
c2.metric("Days collected", df["date"].nunique())
c3.metric("Data sources", 3)
top_repo = latest.sort_values("buzz_score", ascending=False).iloc[0]
c4.metric("Top buzz score", f"{top_repo['buzz_score']:.3f}", top_repo["full_name"])

tab0, tab1, tab2, tab3, tab4 = st.tabs(["Pipeline", "Ranking & Trend", "Correlation", "Prediction Model", "Raw Data"])

# ---------- helpers for the Pipeline tab ----------
PIPE_NODES = ["gh", "so", "hn", "ext", "raw", "tr", "load", "db", "fm", "dash"]

# Which nodes are "working" in each stage of run_pipeline.py
STAGE_ACTIVE = {
    1: ["gh", "ext", "raw"],
    2: ["so", "ext", "raw"],
    3: ["hn", "ext", "raw"],
    4: ["tr"],
    5: ["load", "db"],
    6: [],
}
# Text printed by run_pipeline.py that tells us which stage has started
STAGE_MARKERS = [
    ("GitHub ingestion", 1),
    ("Stack Overflow ingestion", 2),
    ("Hacker News ingestion", 3),
    ("Buzz score transformation", 4),
    ("Load buzz_daily", 5),
    ("Pipeline run complete", 6),
]
STAGE_TEXT = {
    0: "Ready. Click a button to start.",
    1: "Stage 1/5: extracting GitHub stars into data/raw/",
    2: "Stage 2/5: extracting Stack Exchange mentions into data/raw/",
    3: "Stage 3/5: extracting Hacker News data into data/raw/",
    4: "Stage 4/5: transforming raw JSON into buzz_scores.csv",
    5: "Stage 5/5: loading today's rows into the SQLite database",
    6: "Complete: new data is in the database and the dashboard can read it.",
}


def node_states(stage):
    """Return {node: 'idle'|'active'|'done'} for a given pipeline stage (0-6)."""
    states = {n: "idle" for n in PIPE_NODES}
    if stage == 0:
        return states
    for k in range(1, stage):                       # earlier stages are finished
        for n in STAGE_ACTIVE[k]:
            states[n] = "done"
    if stage < 6:
        for n in STAGE_ACTIVE[stage]:
            states[n] = "active"
    else:                                           # whole run finished
        for n in PIPE_NODES:
            if n != "fm":
                states[n] = "done"
    return states


def pipeline_dot(stage=0):
    st_ = node_states(stage)

    def node(nid, label, shape="box", store=False, main=False):
        status = st_[nid]
        if status == "active":
            fill, line, pw, font = "#FFC857", "#C77700", 3, "#1F2E4A"
        elif status == "done":
            fill, line, pw, font = "#B7E4C7", "#2D6A4F", 2, "#1F2E4A"
        elif main:
            fill, line, pw, font = "#1F8A8C", "#14696B", 1, "white"
        elif store:
            fill, line, pw, font = "#FFF4E5", "#B5642E", 1, "#1F2E4A"
        else:
            fill, line, pw, font = "#E6F4F4", "#1F8A8C", 1, "#1F2E4A"
        style = "filled" if shape in ("cylinder", "folder", "note") else "rounded,filled"
        return (f'{nid} [label="{label}", shape={shape}, style="{style}", fillcolor="{fill}", '
                f'color="{line}", penwidth={pw}, fontcolor="{font}"];')

    def cluster(cid, title, body):
        return (f'subgraph cluster_{cid} {{ label="{title}"; fontname="Helvetica-Bold"; fontsize=12; '
                f'color="#B0BEC5"; style="rounded"; {body} }}')

    parts = [
        'digraph G { rankdir=LR; nodesep=0.25; ranksep=0.45; bgcolor="transparent";',
        'node [fontname="Helvetica", fontsize=12, margin="0.12,0.07"];',
        'edge [color="#5A6B7B", arrowsize=0.8];',
        cluster("src", "1. SOURCES",
                node("gh", "GitHub API\\n(stars)") +
                node("so", "Stack Exchange\\n(mentions)") +
                node("hn", "Hacker News\\n(mentions, points)")),
        cluster("ext", "2. EXTRACTION",
                node("ext", "ingestion\\nscripts") +
                node("raw", "data/raw/\\n(JSON files)", shape="folder", store=True)),
        cluster("tr", "3. TRANSFORM",
                node("tr", "buzz_score.py\\n(pandas)\\n-> buzz_scores.csv")),
        cluster("st", "4. STORAGE",
                node("load", "load_buzz_daily.py\\n(upsert)") +
                node("db", "SQLite\\ndevbuzz.db", shape="cylinder", store=True)),
        node("fm", "features + model\\n(run separately)"),
        node("dash", "5. DASHBOARD\\n(Streamlit)", main=True),
        "gh -> ext; so -> ext; hn -> ext; ext -> raw -> tr -> load -> db;",
        "db -> dash; db -> fm [style=dashed]; fm -> dash [style=dashed, constraint=false];",
        "{ rank=same; fm; dash; } }",
    ]
    return "\n".join(parts)


def run_stages(mode, diagram_slot, status_slot, bar, log_slot):
    """Run the pipeline (live) or replay it (no API calls), updating the UI as it goes.
    Returns (ok: bool, log_lines: list[str])."""
    log = []
    stage = 0

    def show(stage_now):
        diagram_slot.graphviz_chart(pipeline_dot(stage_now), use_container_width=True)
        status_slot.info(STAGE_TEXT[stage_now])
        bar.progress(stage_now / 6)
        log_slot.code("\n".join(log[-18:]) or "(waiting for output...)", language="text")

    show(0)

    if mode == "replay":
        # Re-plays the stages using the numbers already stored in the database. No API calls.
        rows = latest.sort_values("full_name")
        plan = [
            (1, [f"GitHub: {r.full_name} -> stars={int(r.stars):,}" for r in rows.itertuples()]),
            (2, [f"Stack Exchange: {r.full_name} -> {int(r.so_mentions)} mentions" for r in rows.itertuples()]),
            (3, [f"Hacker News: {r.full_name} -> total_hits={int(r.hn_mentions):,}" for r in rows.itertuples()]),
            (4, [f"{r.full_name} -> buzz_score={r.buzz_score:.4f}" for r in rows.itertuples()]),
            (5, [f"buzz_daily: upserted {len(rows)} rows for {df['date'].max()}"]),
            (6, ["=== Replay complete (no API calls were made) ==="]),
        ]
        for stg, lines in plan:
            stage = stg
            log.append(f"--- {STAGE_TEXT[stg]} ---")
            show(stage)
            for line in lines:
                log.append("  " + line)
                log_slot.code("\n".join(log[-18:]), language="text")
                time.sleep(0.12)
            time.sleep(0.4)
        return True, log

    # ----- live mode: really run src/run_pipeline.py and follow its output -----
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    proc = subprocess.Popen(
        [sys.executable, "-u", os.path.join("src", "run_pipeline.py")],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        encoding="utf-8", errors="replace", bufsize=1, env=env, cwd=os.getcwd(),
    )
    for raw_line in proc.stdout:
        line = raw_line.rstrip()
        if not line:
            continue
        log.append(line)
        for marker, stg in STAGE_MARKERS:
            if marker in line and stg > stage:
                stage = stg
                break
        show(stage)
    code = proc.wait()
    ok = (code == 0 and stage == 6)
    return ok, log


# ---------- the tab itself ----------
with tab0:
    st.subheader("Run the pipeline and watch the data move")
    st.caption("The diagram lights up as data moves from the sources to the database. "
               "Orange = working now, green = finished.")

    b1, b2, _ = st.columns([1, 1, 2])
    run_live = b1.button("Run pipeline (live)", type="primary", use_container_width=True,
                         help="Runs src/run_pipeline.py for real: calls the three APIs, rebuilds the score "
                              "and writes today's rows to the database.")
    run_replay = b2.button("Replay last run (no API calls)", use_container_width=True,
                           help="Animates the same stages using data already stored in the database.")
    st.caption("Live run calls the real APIs (Stack Exchange allows 300 requests per day). "
               "Running it again on the same day is safe: today's rows are updated, never duplicated. "
               "Use the replay if you only want to demonstrate the flow.")

    diagram_slot = st.empty()
    status_slot = st.empty()
    bar = st.progress(0)
    log_slot = st.empty()

    if run_live or run_replay:
        mode = "live" if run_live else "replay"
        ok, log = run_stages(mode, diagram_slot, status_slot, bar, log_slot)
        if ok:
            status_slot.success(STAGE_TEXT[6] + (" (replay)" if mode == "replay" else ""))
            if mode == "live":
                st.cache_data.clear()   # next page load re-reads the updated database
                st.success("Database updated. Switch tabs or press R to reload the charts with the new data.")
        else:
            status_slot.error("The run did not finish. See the log below.")
            st.code("\n".join(log[-30:]), language="text")
    else:
        diagram_slot.graphviz_chart(pipeline_dot(0), use_container_width=True)
        status_slot.info(STAGE_TEXT[0])

    # --- Live numbers from each stage ---
    st.subheader("What each stage holds right now")
    raw_dir = "data/raw"
    raw_files = 0
    if os.path.exists(raw_dir):
        for _root, _dirs, _files in os.walk(raw_dir):
            raw_files += sum(1 for f in _files if f.endswith(".json"))
    feat_rows = len(load_features())

    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("2. Raw JSON files", raw_files if raw_files else "n/a",
              help="Untouched API responses saved by the ingestion scripts (3 sources x 10 repos). "
                   "Not stored in the GitHub repo; regenerated on every run.")
    s2.metric("3. Repos scored", latest["full_name"].nunique(),
              help="Repos that received a buzz_score in the latest run.")
    s3.metric("4. Rows in database", len(df),
              help="One row per repo per day collected in buzz_daily.")
    s4.metric("5. Model feature rows", feat_rows,
              help="One row per repo, engineered from the full history.")
    s5.metric("Last pipeline run", str(df["date"].max()))

    # --- Stage-by-stage table ---
    st.subheader("Stage by stage")
    stages = pd.DataFrame([
        ["1. Sources", "GitHub, Stack Exchange, Hacker News (public APIs)", "-", "Live web data"],
        ["2. Extraction", "Calls each API once per repo; saves the full response untouched",
         "10 repo names", "data/raw/*.json"],
        ["3. Transformation", "Reads raw JSON, min-max normalises each metric to 0-1, "
         "combines into buzz_score = 0.5 x stars + 0.25 x Stack Exchange + 0.25 x Hacker News",
         "data/raw/*.json", "data/processed/buzz_scores.csv"],
        ["4. Storage", "Upserts today's row for each repo (re-running a day never duplicates)",
         "buzz_scores.csv", "db/devbuzz.db (buzz_daily)"],
        ["5. Analysis & model", "Builds trend features, runs correlations, trains and evaluates the model",
         "buzz_daily history", "model_features.csv, charts, accuracy"],
        ["6. Dashboard", "Reads the database and feature file live and draws everything you see",
         "devbuzz.db, model_features.csv", "These tabs"],
    ], columns=["Stage", "What happens", "Input", "Output"])
    st.dataframe(stages, use_container_width=True, hide_index=True)

    # --- Follow one repo through the pipeline ---
    st.subheader("Follow one repo through the pipeline")
    pick = st.selectbox("Choose a repo", sorted(latest["full_name"].unique()), key="pipeline_repo")
    r = latest[latest["full_name"] == pick].iloc[0]
    rank = int((latest["buzz_score"] > r["buzz_score"]).sum()) + 1

    # Three boxes with identical structure: a heading, then bullet lines in the same style.
    c_a, c_arrow1, c_b, c_arrow2, c_c = st.columns([10, 1, 10, 1, 10])
    with c_a.container(border=True):
        st.markdown(
            "##### 1. Extraction\n"
            "Raw numbers pulled from the sources:\n"
            f"- GitHub stars: **{int(r['stars']):,}**\n"
            f"- Stack Exchange mentions: **{int(r['so_mentions'])}**\n"
            f"- Hacker News mentions: **{int(r['hn_mentions']):,}**\n"
            f"- Hacker News points: **{int(r['hn_points']):,}**"
        )
    c_arrow1.markdown("<div style='text-align:center; font-size:1.8rem; padding-top:3.5rem'>&rarr;</div>",
                      unsafe_allow_html=True)
    with c_b.container(border=True):
        st.markdown(
            "##### 2. Transformation\n"
            "Combined into one score:\n"
            "- Each metric rescaled to 0-1 across all repos\n"
            "- Weighted 50% stars, 25% Stack Exchange, 25% Hacker News\n"
            f"- buzz_score: **{r['buzz_score']:.4f}**"
        )
    c_arrow2.markdown("<div style='text-align:center; font-size:1.8rem; padding-top:3.5rem'>&rarr;</div>",
                      unsafe_allow_html=True)
    with c_c.container(border=True):
        st.markdown(
            "##### 3. Storage and dashboard\n"
            "Kept and shown:\n"
            f"- Saved in `buzz_daily` as a row dated **{r['date']}**\n"
            f"- Rank in the Ranking tab: **#{rank} of {len(latest)}**"
        )
    st.caption("These values are the latest row for the chosen repo, read from the database.")

# --- Tab 1: Ranking + Trend ---
with tab1:
    st.subheader("Current buzz score ranking")
    ranking = latest.sort_values("buzz_score", ascending=False)
    st.bar_chart(ranking.set_index("full_name")["buzz_score"])
    st.caption("Snapshot of each repo's most recent buzz_score — taller bar means more buzz right now.")

    st.subheader("Buzz score trend over time")
    trend = (
        df[df["full_name"].isin(selected_repos)]
        .pivot(index="date", columns="full_name", values="buzz_score")
    )

    fig_trend = go.Figure()
    for col in trend.columns:
        fig_trend.add_trace(go.Scatter(x=trend.index, y=trend[col], mode="lines", name=col))

    # Auto-detect the sharpest decline between the latest two data points and label it —
    # computed from the data each run, never hardcoded to a specific repo.
    if len(trend) >= 2:
        deltas = trend.iloc[-1] - trend.iloc[-2]
        deltas = deltas.dropna()
        if not deltas.empty:
            biggest_drop_repo = deltas.idxmin()
            biggest_drop_value = deltas.min()
            if biggest_drop_value < -0.02:  # only annotate a meaningfully sized drop, not noise
                fig_trend.add_annotation(
                    x=trend.index[-1], y=trend[biggest_drop_repo].iloc[-1],
                    text=f"{biggest_drop_repo} dipped {biggest_drop_value:+.3f}",
                    showarrow=True, arrowhead=2, ay=-40,
                )

    fig_trend.update_layout(
        legend=dict(orientation="h", yanchor="top", y=-0.25, xanchor="center", x=0.5),
        margin=dict(b=90),
        height=450,
        xaxis_title="Date",
        yaxis_title="Buzz Score",
    )
    st.plotly_chart(fig_trend, use_container_width=True)
    st.caption("Each line tracks one repo's buzz_score across every day of real pipeline data collected so far. "
               "Hover any point for its exact value and date.")

# --- Tab 2: Correlation ---
with tab2:
    st.subheader("Correlation heatmap — all metrics")
    corr_cols = {
        "stars": "Stars",
        "buzz_score": "Buzz Score",
        "hn_mentions": "HN Mentions",
        "hn_points": "HN Points",
        "so_mentions": "SO Mentions",
    }
    corr_matrix = latest[list(corr_cols.keys())].rename(columns=corr_cols).corr()

    fig_h, ax_h = plt.subplots(figsize=(5, 4))
    im = ax_h.imshow(corr_matrix, cmap="RdYlGn", vmin=-1, vmax=1)
    ax_h.set_xticks(range(len(corr_matrix.columns)))
    ax_h.set_yticks(range(len(corr_matrix.columns)))
    ax_h.set_xticklabels(corr_matrix.columns, rotation=45, ha="right")
    ax_h.set_yticklabels(corr_matrix.columns)
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            ax_h.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}", ha="center", va="center",
                      color="black", fontsize=9)
    fig_h.colorbar(im, ax=ax_h, label="Pearson r")
    st.pyplot(fig_h)
    st.caption("Every metric correlated against every other metric at once — greener means a stronger "
               "positive relationship, redder means a stronger negative one.")

    st.subheader("Stars vs. each metric")
    col1, col2, col3 = st.columns(3)

    def scatter_vs_stars(ax, y_col, y_label):
        ax.scatter(latest["stars"], latest[y_col])
        for _, row in latest.iterrows():
            ax.annotate(row["name"], (row["stars"], row[y_col]), fontsize=7)
        ax.set_xscale("log")
        ax.set_xlabel("Stars (log)")
        ax.set_ylabel(y_label)

    with col1:
        fig1, ax1 = plt.subplots(figsize=(3.5, 3))
        scatter_vs_stars(ax1, "buzz_score", "Buzz Score")
        st.pyplot(fig1)
        r1 = latest["stars"].corr(latest["buzz_score"])
        st.caption(f"r = {r1:.3f} — strong, but partly circular")

    with col2:
        fig2, ax2 = plt.subplots(figsize=(3.5, 3))
        scatter_vs_stars(ax2, "hn_mentions", "HN Mentions")
        st.pyplot(fig2)
        r2 = latest["stars"].corr(latest["hn_mentions"])
        st.caption(f"r = {r2:.3f} — moderate, independent")

    with col3:
        fig3, ax3 = plt.subplots(figsize=(3.5, 3))
        scatter_vs_stars(ax3, "hn_points", "HN Points")
        st.pyplot(fig3)
        r3 = latest["stars"].corr(latest["hn_points"])
        st.caption(f"r = {r3:.3f} — weak, independent (most interesting finding)")

    st.subheader("Correlation strength, side by side")
    r_compare = pd.DataFrame({
        "relationship": ["Stars vs. buzz_score", "Stars vs. HN mentions", "Stars vs. HN points"],
        "pearson_r": [r1, r2, r3],
    }).set_index("relationship")
    st.bar_chart(r_compare)
    st.caption("The weak bar (HN points) is the project's key finding: how viral a discussion goes is "
               "largely independent of a repo's existing popularity.")

# --- Tab 3: Prediction Model (coefficients of the final model) ---
with tab3:
    feat_df = load_features().dropna(subset=["still_trending"])

    FEATURES = ["latest_stars", "latest_buzz_score", "avg_buzz_score",
                "avg_hn_mentions", "star_growth_rate", "buzz_score_delta"]
    X = feat_df[FEATURES].fillna(0).values
    y = feat_df["still_trending"].astype(int).values

    final_scaler = StandardScaler()
    final_model = LogisticRegression(max_iter=1000).fit(final_scaler.fit_transform(X), y)
    coef_df = pd.DataFrame({"feature": FEATURES, "coefficient": final_model.coef_[0]}).sort_values("coefficient")
    st.subheader("Feature coefficients")
    st.bar_chart(coef_df.set_index("feature"))
    st.caption("Standardized coefficients from the final model (trained on all repos) — positive bars push "
               "the prediction toward 'trending', negative bars push away from it.")
    # st.caption("Model evaluation (Leave-One-Out cross-validation against a majority-class baseline) is run by "
    #            "src/model/train_model.py and reported in the project report.")

# --- Tab 4: Raw Data ---
with tab4:
    st.subheader("buzz_daily (joined with repos)")
    st.dataframe(df.drop(columns=["forks", "open_issues"]), use_container_width=True)
    st.caption("Every row collected by the daily pipeline so far, joined with repo names for readability.")
    st.download_button("Download as CSV", df.to_csv(index=False), "buzz_daily_export.csv")