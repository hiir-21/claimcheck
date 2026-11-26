"""
V3: the fine-tuned checker judges the evidence, with no LLM at all.

For each exam claim, we reuse the evidence V2.1 already retrieved (saved in
results/v2_hybrid_predictions.jsonl), so the ONLY difference from V2.1 is who judges it:
the LLM (V2.1) or our own trained model (V3). No API calls, so no daily limit.

Run:  python src/checker.py
Then: python src/evaluate.py results/v3_predictions.jsonl --name V3-checker
"""

import argparse
import json
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
MAX_LEN = 256


def evidence_text(evidence):
    """Same format as the training data: 'Title . sentence Title . sentence ...'"""
    if not evidence:
        return "No evidence found."
    return " ".join(f"{ev['title']} . {ev['sentence']}" for ev in evidence)


class Checker:
    def __init__(self, model_id):
        print(f"Loading checker: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).eval()
        self.labels = self.model.config.id2label

    def predict(self, claims, evidence_texts):
        """Returns, for each pair, the best label and the probability of every label."""
        enc = self.tokenizer(claims, evidence_texts, truncation="only_second", max_length=MAX_LEN,
                             padding=True, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(self.model(**enc).logits, dim=-1).tolist()
        results = []
        for p in probs:
            best = max(range(len(p)), key=lambda i: p[i])
            results.append((self.labels[best], {self.labels[i]: round(x, 4) for i, x in enumerate(p)}))
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("CHECKER_MODEL"),
                        help="Hugging Face model id, e.g. your-username/claimcheck-deberta-v3-fever")
    parser.add_argument("--evidence", default="results/v2_hybrid_predictions.jsonl",
                        help="predictions file whose saved evidence we reuse")
    parser.add_argument("--out", default="results/v3_predictions.jsonl")
    args = parser.parse_args()
    if not args.model:
        raise SystemExit("Set CHECKER_MODEL in .env or pass --model your-username/claimcheck-deberta-v3-fever")

    claims = {json.loads(l)["id"]: json.loads(l)["claim"] for l in open(ROOT / "data" / "fever_test.jsonl", encoding="utf-8")}
    rows = [json.loads(l) for l in open(ROOT / args.evidence, encoding="utf-8")]
    checker = Checker(args.model)

    out_path = ROOT / args.out
    with open(out_path, "w", encoding="utf-8") as out:
        for i in tqdm(range(0, len(rows), 16)):  # 16 claims at a time
            batch = rows[i:i + 16]
            preds = checker.predict([claims[r["id"]] for r in batch], [evidence_text(r["evidence"]) for r in batch])
            for r, (label, probs) in zip(batch, preds):
                out.write(json.dumps({"id": r["id"], "pred": label, "probs": probs, "evidence": r["evidence"]}) + "\n")
    print(f"Done. Predictions in {out_path}")


if __name__ == "__main__":
    main()
