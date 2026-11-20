"""
Why did V2 get claims wrong: bad retrieval, or bad reasoning?

FEVER tells us which Wikipedia page holds the real evidence for every SUPPORTS and REFUTES
claim. So for those 200 claims we can check two things:
  1. Page found   - did the search find the correct page at all?
  2. In evidence  - did a sentence from that page make it into the top 5 shown to the LLM?
Then we compare accuracy when retrieval worked vs. when it failed.
If accuracy is much higher when the page was found, retrieval is the bottleneck.

Run:  python src/analyze_retrieval.py results/v2_predictions.jsonl
"""

import json
import sys
from collections import Counter
from pathlib import Path

from retriever import find_titles  # uses the saved cache, so no new Wikipedia requests

ROOT = Path(__file__).resolve().parent.parent
LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]


def fever_title(wiki_url):
    """FEVER writes 'Swordfish_-LRB-film-RRB-'; Wikipedia writes 'Swordfish (film)'."""
    t = wiki_url.replace("_", " ")
    for code, char in {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]",
                       "-LCB-": "{", "-RCB-": "}", "-COLON-": ":"}.items():
        t = t.replace(code, char)
    return t


def main():
    pred_file = Path(sys.argv[1])
    gold = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "data" / "fever_test.jsonl", encoding="utf-8")}
    preds = [json.loads(l) for l in open(pred_file, encoding="utf-8")]

    stats = Counter()
    for p in preds:
        g = gold[p["id"]]
        if g["label"] == "NOT ENOUGH INFO":
            continue
        gold_pages = {fever_title(ev[2]) for group in g["evidence"] for ev in group if ev[2]}
        # Newer runs save the pages they searched; for older runs, recompute with the default search.
        searched = set(p["pages"]) if "pages" in p else set(find_titles(g["claim"]))
        shown = {ev["title"] for ev in p["evidence"]}
        found = bool(gold_pages & searched)
        in_ev = bool(gold_pages & shown)
        correct = p["pred"] == g["label"]
        stats["total"] += 1
        stats["found"] += found
        stats["in_ev"] += in_ev
        stats[f"correct_found_{found}"] += correct
        stats[f"n_found_{found}"] += 1

    n = stats["total"]
    print(f"\nRetrieval check on {n} SUPPORTS/REFUTES claims ({pred_file.name})")
    print(f"  Correct page found by search : {stats['found']}/{n} ({stats['found'] / n:.0%})")
    print(f"  Correct page in top-5 shown  : {stats['in_ev']}/{n} ({stats['in_ev'] / n:.0%})")
    for found in (True, False):
        k = stats[f"n_found_{found}"]
        if k:
            acc = stats[f"correct_found_{found}"] / k
            label = "page FOUND    " if found else "page NOT found"
            print(f"  Accuracy when {label}: {acc:.0%} ({k} claims)")

    # Confusion matrix: rows = correct label, columns = what the system said
    print("\nConfusion matrix (rows = correct label, columns = predicted)")
    cm = Counter((gold[p["id"]]["label"], p["pred"]) for p in preds)
    print(f"{'':>18}" + "".join(f"{l[:12]:>14}" for l in LABELS))
    for g in LABELS:
        print(f"{g:>18}" + "".join(f"{cm[(g, pr)]:>14}" for pr in LABELS))


if __name__ == "__main__":
    main()
