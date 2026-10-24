import os
import json
import time
import requests

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

BASE_URL = "https://hn.algolia.com/api/v1/search"

def search_hn_mentions(query: str, hits_per_page: int = 100) -> dict:
    params = {
        "query": query,
        "tags": "story",
        "hitsPerPage": hits_per_page,
    }
    response = requests.get(BASE_URL, params=params)
    response.raise_for_status()
    data = response.json()

    hits = [
        {
            "title": hit.get("title"),
            "points": hit.get("points"),
            "num_comments": hit.get("num_comments"),
            "created_at": hit.get("created_at"),
            "url": hit.get("url"),
            "hn_object_id": hit.get("objectID"),
        }
        for hit in data.get("hits", [])
    ]

    return {
        "total_hits": data.get("nbHits", len(hits)),  # TRUE total match count
        "hits": hits,
    }

if __name__ == "__main__":
    os.makedirs("data/raw/hackernews", exist_ok=True)

    for full_name, query in TRACKED_REPOS.items():
        print(f"Searching Hacker News for: {query} ...")
        result = search_hn_mentions(query)

        safe_filename = full_name.replace("/", "_")
        filepath = f"data/raw/hackernews/{safe_filename}.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print(f"  -> total_hits={result['total_hits']}, saved {len(result['hits'])} detailed records")