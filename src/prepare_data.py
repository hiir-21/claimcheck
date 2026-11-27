"""
Step 1: download FEVER and build a small, balanced test set.

What it does:
1. Downloads FEVER's official "paper_dev" file (about 10,000 labelled claims).
2. Picks an equal number of claims from each label (SUPPORTS, REFUTES, NOT ENOUGH INFO).
3. Saves them to data/fever_test.jsonl, one claim per line.

Run:  python src/prepare_data.py
      python src/prepare_data.py --per-label 150   (450 claims instead of 300)
      Dev set for V3.1 (150 different claims, never the exam ones):
      python src/prepare_data.py --per-label 50 --seed 7 --exclude data/fever_test.jsonl --out data/fever_dev.jsonl
"""

import argparse
import json
import random
import urllib.request
from collections import Counter
from pathlib import Path

URL = "https://fever.ai/download/fever/paper_dev.jsonl"
ROOT = Path(__file__).resolve().parent.parent
RAW_FILE = ROOT / "data" / "raw" / "paper_dev.jsonl"
OUT_FILE = ROOT / "data" / "fever_test.jsonl"
LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]


def download():
    """Download the FEVER file once; skip if we already have it."""
    if RAW_FILE.exists():
        print(f"Already downloaded: {RAW_FILE}")
        return
    RAW_FILE.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL} ...")
    urllib.request.urlretrieve(URL, RAW_FILE)
    print(f"Saved to {RAW_FILE}")


def load_claims():
    """Read the JSONL file: each line is one claim with its label."""
    claims = []
    with open(RAW_FILE, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            claims.append(
                {
                    "id": item["id"],
                    "claim": item["claim"],
                    "label": item["label"],
                    # Kept for later steps (which Wikipedia pages the evidence came from)
                    "evidence": item.get("evidence", []),
                }
            )
    return claims


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-label", type=int, default=100, help="claims per label (default 100 -> 300 total)")
    parser.add_argument("--seed", type=int, default=42, help="random seed, so the same claims are picked every time")
    parser.add_argument("--exclude", default=None, help="a JSONL file whose claims must NOT be picked (e.g. the exam)")
    parser.add_argument("--out", default=str(OUT_FILE), help="where to save the chosen claims")
    args = parser.parse_args()

    download()
    claims = load_claims()
    print(f"Loaded {len(claims)} claims. Label counts: {dict(Counter(c['label'] for c in claims))}")

    if args.exclude:
        excluded = {json.loads(l)["id"] for l in open(args.exclude, encoding="utf-8")}
        claims = [c for c in claims if c["id"] not in excluded]
        print(f"Excluded {len(excluded)} claims from {args.exclude}; {len(claims)} left to choose from.")

    random.seed(args.seed)
    test_set = []
    for label in LABELS:
        pool = [c for c in claims if c["label"] == label]
        test_set += random.sample(pool, args.per_label)
    random.shuffle(test_set)

    with open(args.out, "w", encoding="utf-8") as f:
        for c in test_set:
            f.write(json.dumps(c) + "\n")

    print(f"Saved {len(test_set)} claims to {args.out}")
    print("Three examples:")
    for c in test_set[:3]:
        print(f"  [{c['label']}] {c['claim']}")


if __name__ == "__main__":
    main()
