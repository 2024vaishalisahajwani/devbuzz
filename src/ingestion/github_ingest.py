import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()  # reads .env and loads GITHUB_TOKEN into environment variables
TOKEN = os.getenv("GITHUB_TOKEN")

HEADERS = {
    "Authorization": f"Bearer {TOKEN}",
    "Accept": "application/vnd.github+json",
}

TRACKED_REPOS = [
    ("astral-sh", "uv"),
    ("browser-use", "browser-use"),
    ("ollama", "ollama"),
    ("langchain-ai", "langchain"),
    ("vercel", "next.js"),
    ("sveltejs", "svelte"),
    ("tokio-rs", "tokio"),
    ("golang", "go"),
    ("duckdb", "duckdb"),
    ("supabase", "supabase"),
]

def fetch_repo_metadata(owner: str, repo: str) -> dict:
    url = f"https://api.github.com/repos/{owner}/{repo}"
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.json()  # keep the FULL raw response at ingestion stage

def save_raw(owner: str, repo: str, data: dict):
    os.makedirs("data/raw", exist_ok=True)
    filename = f"data/raw/{owner}_{repo}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    for owner, repo in TRACKED_REPOS:
        print(f"Fetching {owner}/{repo} ...")
        data = fetch_repo_metadata(owner, repo)
        save_raw(owner, repo, data)
        print(f"  -> saved, stars={data['stargazers_count']}")

    print("\nDone. Check remaining rate limit:")
    rate = requests.get("https://api.github.com/rate_limit", headers=HEADERS).json()
    print(json.dumps(rate["rate"], indent=2))