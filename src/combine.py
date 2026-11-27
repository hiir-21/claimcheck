"""
V3.1: combine the two judges, the LLM (V2.1) and our fine-tuned checker (V3).

The LLM is cautious (good at NOT ENOUGH INFO, misses false claims); the checker is decisive
(catches false claims, rarely says NOT ENOUGH INFO). Two simple rules for combining them:

  Rule A "confident checker": if the checker is at least T sure, use its answer; otherwise use the LLM's.
  Rule B "LLM first":         if the LLM says SUPPORTS or REFUTES, keep it; if it says NOT ENOUGH INFO,
                              use the checker's answer only if the checker is at least T sure.

IMPORTANT: the rule and the threshold T are chosen on the DEV set only, never on the exam.
  1. tune  -> tries both rules with several thresholds on the dev set, saves the best to results/combine_rule.json
  2. apply -> applies that saved rule ONCE to the exam, without looking at exam scores

Run:  python src/combine.py tune
      python src/combine.py apply
Then: python src/evaluate.py results/v31_predictions.jsonl --name V3.1-combined
"""

import json
import sys
from pathlib import Path

from sklearn.metrics import f1_score

ROOT = Path(__file__).resolve().parent.parent
RULE_FILE = ROOT / "results" / "combine_rule.json"
LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
THRESHOLDS = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95]  # kept short on purpose: fewer choices = less overfitting the dev set


def read(path):
    return {r["id"]: r for r in (json.loads(l) for l in open(ROOT / path, encoding="utf-8"))}


def combine(llm_label, checker_label, checker_conf, rule, t):
    if rule == "A":
        return checker_label if checker_conf >= t else llm_label
    if rule == "B":
        if llm_label != "NOT ENOUGH INFO":
            return llm_label
        return checker_label if checker_conf >= t else "NOT ENOUGH INFO"
    raise ValueError(rule)


def combined_preds(llm, checker, rule, t):
    out = {}
    for cid in llm:
        probs = checker[cid]["probs"]
        out[cid] = combine(llm[cid]["pred"], checker[cid]["pred"], max(probs.values()), rule, t)
    return out


def macro_f1(gold, preds):
    ids = list(gold)
    return f1_score([gold[i] for i in ids], [preds[i] for i in ids], labels=LABELS, average="macro", zero_division=0)


def tune():
    gold = {cid: r["label"] for cid, r in read("data/fever_dev.jsonl").items()}
    llm = read("results/dev_v2_hybrid_predictions.jsonl")
    checker = read("results/dev_v3_predictions.jsonl")
    assert set(gold) == set(llm) == set(checker), "dev files don't cover the same claims: finish both dev runs first"

    print(f"Dev set: {len(gold)} claims")
    print(f"  LLM alone (V2.1)    macro-F1: {macro_f1(gold, {i: llm[i]['pred'] for i in gold}):.3f}")
    print(f"  Checker alone (V3)  macro-F1: {macro_f1(gold, {i: checker[i]['pred'] for i in gold}):.3f}\n")

    best = None
    for rule in ("A", "B"):
        for t in THRESHOLDS:
            score = macro_f1(gold, combined_preds(llm, checker, rule, t))
            print(f"  Rule {rule}, T = {t:.2f}: macro-F1 {score:.3f}")
            if best is None or score > best["dev_macro_f1"]:
                best = {"rule": rule, "threshold": t, "dev_macro_f1": round(score, 4)}
    RULE_FILE.write_text(json.dumps(best, indent=2))
    print(f"\nBest on dev: Rule {best['rule']}, T = {best['threshold']} (macro-F1 {best['dev_macro_f1']}). Saved to {RULE_FILE.name}.")
    print("Now run:  python src/combine.py apply")


def apply():
    rule = json.loads(RULE_FILE.read_text())
    llm = read("results/v2_hybrid_predictions.jsonl")
    checker = read("results/v3_predictions.jsonl")
    preds = combined_preds(llm, checker, rule["rule"], rule["threshold"])
    out = ROOT / "results" / "v31_predictions.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for cid, label in preds.items():
            f.write(json.dumps({"id": cid, "pred": label, "llm": llm[cid]["pred"], "checker": checker[cid]["pred"],
                                "checker_conf": max(checker[cid]["probs"].values()),
                                "pages": llm[cid].get("pages", []), "evidence": llm[cid]["evidence"]}) + "\n")
    print(f"Applied Rule {rule['rule']}, T = {rule['threshold']} (chosen on dev) to the exam: {out}")
    print("Now run:  python src/evaluate.py results/v31_predictions.jsonl --name V3.1-combined")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "tune":
        tune()
    elif mode == "apply":
        apply()
    else:
        print(__doc__)
