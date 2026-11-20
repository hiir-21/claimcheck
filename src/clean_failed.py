"""
Removes claims whose Wikipedia lookup failed, so they can be run again.

When the connection to Wikipedia drops, a claim can end up with no evidence at all.
The model then (correctly) says NOT ENOUGH INFO, but that's unfair to V2: the problem
was the internet, not the method. This script:
1. removes predictions that have an empty evidence list from the predictions file, and
2. removes empty entries from the Wikipedia cache,
so that running rag_llm.py again redoes just those claims.

Run:  python src/clean_failed.py results/v2_predictions.jsonl
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE_FILE = ROOT / "data" / "wiki_cache.json"

pred_file = Path(sys.argv[1])
rows = [json.loads(line) for line in open(pred_file, encoding="utf-8")]
kept = [r for r in rows if r.get("evidence")]
with open(pred_file, "w", encoding="utf-8") as f:
    for r in kept:
        f.write(json.dumps(r) + "\n")
print(f"{pred_file.name}: removed {len(rows) - len(kept)} claims with no evidence, kept {len(kept)}.")

if CACHE_FILE.exists():
    cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    before = len(cache["search"]) + len(cache["pages"])
    cache["search"] = {k: v for k, v in cache["search"].items() if v}
    cache["pages"] = {k: v for k, v in cache["pages"].items() if v}
    after = len(cache["search"]) + len(cache["pages"])
    CACHE_FILE.write_text(json.dumps(cache), encoding="utf-8")
    print(f"Wikipedia cache: removed {before - after} empty entries.")
