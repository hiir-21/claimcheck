"""
Quick test of the page search alone: no LLM, no ranking, just "did we find the right page?"

Compares the original search (full claim) with the hybrid search (key names + full claim)
on the 200 SUPPORTS/REFUTES claims, using FEVER's record of which page holds the evidence.
It only calls Wikipedia's search, so it takes a few minutes, not 40.

Run:  python src/check_search.py
"""

import json
from pathlib import Path

from tqdm import tqdm

from analyze_retrieval import fever_title  # same title conversion as the analysis
from retriever import find_titles, save_cache

ROOT = Path(__file__).resolve().parent.parent
claims = [json.loads(l) for l in open(ROOT / "data" / "fever_test.jsonl", encoding="utf-8")]
claims = [c for c in claims if c["label"] != "NOT ENOUGH INFO"]

hits = {"claim": 0, "hybrid": 0}
pages = {"claim": 0, "hybrid": 0}
for i, c in enumerate(tqdm(claims), 1):
    gold = {fever_title(ev[2]) for group in c["evidence"] for ev in group if ev[2]}
    for mode in hits:
        found = find_titles(c["claim"], hybrid=(mode == "hybrid"))
        hits[mode] += bool(gold & set(found))
        pages[mode] += len(found)
    if i % 20 == 0:
        save_cache()
save_cache()

n = len(claims)
print(f"\nCorrect page found, out of {n} SUPPORTS/REFUTES claims:")
for mode in hits:
    print(f"  {mode:>6} search: {hits[mode]}/{n} ({hits[mode] / n:.0%}), "
          f"average {pages[mode] / n:.1f} pages read per claim")
