import os
import json
import time
import requests
from dotenv import load_dotenv

load_dotenv()
STACK_KEY = os.getenv("STACK_EXCHANGE_KEY")  # optional — None is fine for this project's scale

TRACKED_REPOS = {
    "astral-sh/uv": "astral uv",
    "browser-use/browser-use": "browser-use",
    "ollama/ollama": "ollama",
    "langchain-ai/langchain": "langchain",
    "vercel/next.js": "next.js",
    "sveltejs/svelte": "svelte",
    "tokio-rs/tokio": "tokio rust",
    "golang/go": "golang",
    "duckdb/duckdb": "duckdb",
    "supabase/supabase": "supabase",
}

BASE_URL = "https://api.stackexchange.com/2.3/search/advanced"

def search_mentions(query: str, pagesize: int = 100) -> list:
    params = {
        "order": "desc",
        "sort": "creation",
        "q": query,
        "site": "stackoverflow",
        "pagesize": pagesize,
    }
    if STACK_KEY:
        params["key"] = STACK_KEY

    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()
    data = response.json()  # requests auto-decompresses the gzip response

    return [
        {
            "title": item["title"],
            "score": item["score"],
            "answer_count": item["answer_count"],
            "view_count": item["view_count"],
            "creation_date": item["creation_date"],
            "tags": item.get("tags", []),
            "link": item["link"],
        }
        for item in data.get("items", [])
    ]

if __name__ == "__main__":
    os.makedirs("data/raw/stackoverflow", exist_ok=True)

    for full_name, query in TRACKED_REPOS.items():
        print(f"Searching Stack Overflow for: {query} ...")
        mentions = search_mentions(query)

        safe_filename = full_name.replace("/", "_")
        filepath = f"data/raw/stackoverflow/{safe_filename}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(mentions, f, indent=2)

        print(f"  -> {len(mentions)} mentions saved")
        time.sleep(0.2)  # small pause between calls — polite API usage