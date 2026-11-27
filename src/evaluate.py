"""
The "exam grader": scores any version of ClaimCheck against the correct answers.

It compares a predictions file (what the system answered) with the test set (the correct labels)
and reports:
- accuracy: share of claims answered correctly
- macro-F1: the average F1 across the three labels, so a system cannot score well
  by always guessing the most common answer
- a 95% confidence interval for both (bootstrap), because the test set is small

Run:  python src/evaluate.py results/v1_predictions.jsonl --name V1
      python src/evaluate.py --self-test     (checks the grader itself, no API needed)
"""

import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, f1_score

ROOT = Path(__file__).resolve().parent.parent
GOLD_FILE = ROOT / "data" / "fever_test.jsonl"
SCORES_FILE = ROOT / "results" / "scores.csv"
LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def bootstrap_ci(gold, pred, metric, n=1000, seed=0):
    """Resample the test set n times with replacement; the middle 95% of scores is the interval."""
    rng = np.random.default_rng(seed)
    gold, pred = np.array(gold), np.array(pred)
    scores = []
    for _ in range(n):
        idx = rng.integers(0, len(gold), len(gold))
        scores.append(metric(gold[idx], pred[idx]))
    return np.percentile(scores, 2.5), np.percentile(scores, 97.5)


def macro_f1(g, p):
    return f1_score(g, p, labels=LABELS, average="macro", zero_division=0)


def score(gold_labels, pred_labels):
    acc = accuracy_score(gold_labels, pred_labels)
    mf1 = macro_f1(gold_labels, pred_labels)
    acc_ci = bootstrap_ci(gold_labels, pred_labels, accuracy_score)
    f1_ci = bootstrap_ci(gold_labels, pred_labels, macro_f1)
    return acc, acc_ci, mf1, f1_ci


def evaluate(pred_file, name, gold_file=GOLD_FILE, save=True):
    gold = {c["id"]: c["label"] for c in read_jsonl(gold_file)}
    preds = {p["id"]: p["pred"] for p in read_jsonl(pred_file)}

    missing = [i for i in gold if i not in preds]
    if missing:
        print(f"Warning: {len(missing)} claims have no prediction; they count as wrong.")

    ids = list(gold)
    g = [gold[i] for i in ids]
    p = [preds.get(i, "MISSING") for i in ids]

    acc, acc_ci, mf1, f1_ci = score(g, p)
    print(f"\n=== {name}: {len(ids)} claims ===")
    print(f"Accuracy : {acc:.3f}  (95% CI {acc_ci[0]:.3f} to {acc_ci[1]:.3f})")
    print(f"Macro-F1 : {mf1:.3f}  (95% CI {f1_ci[0]:.3f} to {f1_ci[1]:.3f})\n")
    print(classification_report(g, p, labels=LABELS, zero_division=0))

    if save:
        SCORES_FILE.parent.mkdir(exist_ok=True)
        new = not SCORES_FILE.exists()
        with open(SCORES_FILE, "a", newline="") as f:
            w = csv.writer(f)
            if new:
                w.writerow(["timestamp", "version", "n", "accuracy", "acc_low", "acc_high", "macro_f1", "f1_low", "f1_high"])
            w.writerow([datetime.now().isoformat(timespec="seconds"), name, len(ids),
                        round(acc, 4), round(acc_ci[0], 4), round(acc_ci[1], 4),
                        round(mf1, 4), round(f1_ci[0], 4), round(f1_ci[1], 4)])
        print(f"Saved scores to {SCORES_FILE}")


def self_test():
    """Sanity check: a perfect system should score 1.0, a random guesser about 0.33."""
    random.seed(1)
    gold = [random.choice(LABELS) for _ in range(300)]
    acc, _, mf1, _ = score(gold, gold)
    print(f"Perfect answers -> accuracy {acc:.2f}, macro-F1 {mf1:.2f}  (expect 1.00)")
    guesses = [random.choice(LABELS) for _ in gold]
    acc, _, mf1, _ = score(gold, guesses)
    print(f"Random guesses  -> accuracy {acc:.2f}, macro-F1 {mf1:.2f}  (expect about 0.33)")
    always = ["SUPPORTS"] * len(gold)
    acc, _, mf1, _ = score(gold, always)
    print(f"Always SUPPORTS -> accuracy {acc:.2f}, macro-F1 {mf1:.2f}  (macro-F1 punishes this)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("pred_file", nargs="?", help="predictions JSONL with fields id and pred")
    parser.add_argument("--name", default="unnamed", help="version name for the scores table, e.g. V1")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--gold", default=str(GOLD_FILE), help="file with the correct labels (default: the exam)")
    args = parser.parse_args()

    if args.self_test:
        self_test()
    elif args.pred_file:
        evaluate(args.pred_file, args.name, gold_file=args.gold)
    else:
        parser.print_help()
