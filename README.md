# DevBuzz: Open-Source Trend & Community Engagement Pipeline

A Python data-engineering pipeline that measures the *current community buzz* around 10 popular open-source projects. It collects data every day from three public sources, combines it into one comparable score, stores the history in a database, analyses it, trains a small prediction model, and shows everything in a live dashboard.

## Why this project

GitHub stars show how popular a project *has been*, but not whether people are discussing it *today*. DevBuzz combines three signals into one daily **buzz score** so projects can be compared and tracked over time.

## Tracked repositories

`astral-sh/uv` · `browser-use/browser-use` · `ollama/ollama` · `langchain-ai/langchain` · `vercel/next.js` · `sveltejs/svelte` · `tokio-rs/tokio` · `golang/go` · `duckdb/duckdb` · `supabase/supabase`

## Architecture

```
GitHub API      Stack Exchange API     Hacker News (Algolia)
     \                  |                      /
      \_________________v_____________________/
                   EXTRACTION
         (raw JSON saved per source, per repo)
                        |
                        v
                  TRANSFORMATION
      (pandas: min-max normalise -> weighted buzz_score)
                        |
                        v
        TARGET STORAGE (SQLite: repos + buzz_daily, upsert)
                        |
           _____________|_____________
          |                           |
          v                           v
   ANALYTICS (EDA)           FEATURE ENGINEERING
   ranking / scatter /       trend slope, growth rate
   trend charts                       |
                                      v
                          PREDICTION MODEL (LOOCV)
                       logistic regression -> still_trending
                                      |
                                      v
                        LIVE DASHBOARD (Streamlit)
```

## Buzz score

`buzz_score = 0.5 x norm(stars) + 0.25 x so_activity + 0.25 x hn_activity`

Each metric is min-max normalised to 0-1 across the tracked repos, so metrics on very different scales (stars in the 100,000s vs. points in the 100s) can be combined fairly.

## Tech stack

| Layer | Tool |
|---|---|
| Language | Python 3 |
| Data collection | `requests`, GitHub REST API, Stack Exchange API, Hacker News (Algolia) API |
| Secrets | `python-dotenv` (`.env` file, never committed) |
| Transformation | `pandas`, `numpy` |
| Storage | SQLite (`db/devbuzz.db`) |
| Modelling | `scikit-learn` (logistic regression, Leave-One-Out CV) |
| Charts | `matplotlib`, `plotly` |
| Dashboard | `streamlit` |

## Project structure

```
devbuzz/
├── .streamlit/config.toml     # dashboard theme
├── data/processed/            # cleaned CSVs (buzz_scores, model_features)
├── db/devbuzz.db              # SQLite database (repos + buzz_daily history)
├── reports/                   # DevBuzz_Project_Report (.pdf / .docx) + figures/ (generated charts)
├── src/
│   ├── ingestion/             # github_ingest.py, stackoverflow_ingest.py, hn_search.py
│   ├── transform/             # buzz_score.py
│   ├── storage/               # db_setup.py, load_repos.py, load_buzz_daily.py
│   ├── analytics/             # visualize.py
│   ├── model/                 # build_features.py, train_model.py
│   ├── dashboard/             # app.py (Streamlit)
│   └── run_pipeline.py        # runs ingestion -> transform -> load in one go
├── .env.example               # template for your API keys
├── requirements.txt
└── README.md
```

## Database schema

| Table | Purpose |
|---|---|
| `repos` | Static metadata, one row per tracked repository |
| `buzz_daily` | Daily time series, one row per repo per day (`UNIQUE(repo_id, date)`); loads use an upsert so re-running the same day never duplicates rows |

## How to run

**1. Clone and set up**

```bash
git clone https://github.com/2024vaishalisahajwani/devbuzz.git
cd devbuzz
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

**2. Add your API keys**

Copy `.env.example` to `.env` and fill in:

- `GITHUB_TOKEN` - a GitHub personal access token (read-only is enough)
- `STACK_EXCHANGE_KEY` - a free key from https://stackapps.com

**3. Run the daily pipeline**

```bash
python src\run_pipeline.py
```

This collects fresh data, computes the buzz score, and adds today's rows to the database. The same run can be started from the dashboard's **Pipeline** tab.

**4. Explore the results**

```bash
python src\analytics\visualize.py      # regenerate charts
python src\model\build_features.py     # engineer model features
python src\model\train_model.py        # train + evaluate the model
streamlit run src\dashboard\app.py     # launch the live dashboard
```

The repository already includes a populated `db/devbuzz.db`, so the dashboard works without running the pipeline first (use **Replay last run** in the Pipeline tab). `data/raw/` is not committed; it is regenerated on each pipeline run, so the raw-file count shows `n/a` until you run the pipeline once.

## Dashboard

Under the header there are download buttons for the full project report (PDF and Word). Five tabs:

- **Pipeline** - a left-to-right flow diagram from the three sources to the dashboard. **Run pipeline (live)** executes `src/run_pipeline.py` for real and lights up each stage as it runs (orange = working, green = finished), with a progress bar and live log. **Replay last run (no API calls)** animates the same stages from data already in the database. Below the diagram: live counts for each stage, a stage-by-stage table, and a "Follow one repo through the pipeline" walkthrough.
- **Ranking & Trend** - current buzz score ranking and an interactive trend chart for all repos. Click a repo name in the chart legend to hide it, or double-click to show only that repo.
- **Correlation** - heatmap and stars-vs-metric scatter plots.
- **Prediction Model** - standardised feature coefficients of the final logistic regression model, plus a per-repo predictions table with CSV download (in-sample, labelled as such). The full evaluation (Leave-One-Out cross-validation against a majority-class baseline) is run by `src/model/train_model.py` and described below.
- **Raw Data** - the joined table with CSV export (the empty `forks` / `open_issues` columns are hidden from the display; see limitations).

> The live run needs your API keys in `.env` and uses real API quota (Stack Exchange allows 300 requests per day). Running it twice on the same day is safe: today's rows are updated, never duplicated. The pipeline run covers extraction, transformation and loading; `build_features.py` and `train_model.py` are run separately.

## Key findings

- **Stars vs. Hacker News points: r = 0.136.** A repo's star count says very little about how viral an individual Hacker News discussion gets, so popularity and current buzz are largely independent signals.
- **Stars vs. buzz_score: r = 0.835**, but this is partly circular because stars make up 50% of the score by construction. The independent correlations above are the more meaningful ones.

## Model result (reported as it is)

A logistic regression predicts whether a repo's buzz score trended upward over the observation window. It was evaluated with Leave-One-Out cross-validation, with scaling fitted inside each fold to avoid data leakage, and `buzz_trend_slope` was excluded from the features because the label is derived from it.

The model scored **100% accuracy, tying (not beating) a majority-class baseline of 100%**. The data has 9 "trending" repos and only 1 "not trending", so no classifier can be shown to outperform simply guessing the majority class. One cross-validation fold was skipped because its training set contained a single class. The coefficients are directionally sensible (`star_growth_rate` and `buzz_score_delta` are the strongest positive predictors), but the result cannot yet be validated at this sample size.

## Known limitations

- **Small sample:** 10 repositories and about 17 days of data, so correlations and the model are exploratory, not statistically robust.
- **Class imbalance:** only one negative example for the model to learn from.
- **Stack Exchange counts** are capped at the API page size (100), a documented scope limitation.
- **Collection gaps:** the pipeline was not run on every calendar day, so the time series has irregular spacing. This is shown as-is in the charts rather than hidden.
- **`forks` and `open_issues`** exist in the `buzz_daily` schema but are not populated by the daily load, so they are hidden from the dashboard's Raw Data table (they are still in the database and CSV export). They are not used by the score or the model. GitHub only returns current counts, so earlier days could not be backfilled.
- **Error handling** uses a fail-loudly approach (`raise_for_status`) with no automatic retry. Retry with backoff would be the first reliability upgrade.
- **Storage:** SQLite was chosen over S3/GCS because the dataset is a few hundred KB and every consumer needs SQL joins. At larger scale the natural path is raw JSON in a date-partitioned bucket, Parquet for curated data, and a managed warehouse.

## Future work

Track hundreds of repositories (including some in decline), run the pipeline on a schedule, add retry logic, populate `forks` / `open_issues`, and re-evaluate the model on a longer history.
