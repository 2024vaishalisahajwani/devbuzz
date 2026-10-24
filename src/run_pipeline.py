import subprocess
import sys
from datetime import date

STEPS = [
    ("GitHub ingestion",           ["python", "src/ingestion/github_ingest.py"]),
    ("Stack Overflow ingestion",   ["python", "src/ingestion/stackoverflow_ingest.py"]),
    ("Hacker News ingestion",      ["python", "src/ingestion/hn_search.py"]),
    ("Buzz score transformation",  ["python", "src/transform/buzz_score.py"]),
    ("Load buzz_daily",            ["python", "src/storage/load_buzz_daily.py"]),
]

def run_pipeline():
    print(f"=== DevBuzz Pipeline Run: {date.today().isoformat()} ===\n")
    for step_name, command in STEPS:
        print(f"--- {step_name} ---")
        result = subprocess.run(command)
        if result.returncode != 0:
            print(f"\n[FAILED] {step_name} exited with code {result.returncode}. Stopping pipeline.")
            sys.exit(1)
        print()
    print("=== Pipeline run complete. ===")

if __name__ == "__main__":
    run_pipeline()