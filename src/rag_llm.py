"""
V2: the LLM answers using evidence retrieved from Wikipedia (RAG).

Same test set, same LLM and same grader as V1. The only change: before asking the LLM,
we retrieve 5 evidence sentences and tell it to decide from those, not from memory.

Run:  python src/rag_llm.py --limit 10            (quick test)
      python src/rag_llm.py                       (V2: embeddings only)
      python src/rag_llm.py --rerank              (V2 + reranker, saved to a separate file)
Then: python src/evaluate.py results/v2_predictions.jsonl --name V2
      python src/evaluate.py results/v2_rerank_predictions.jsonl --name V2+rerank
"""

import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm

from llm import MODEL, ask, parse_label
from retriever import Retriever, save_cache

ROOT = Path(__file__).resolve().parent.parent
TEST_FILE = ROOT / "data" / "fever_test.jsonl"

SYSTEM_PROMPT = """You are a careful fact-checker.
You will be given a claim and numbered evidence sentences retrieved from Wikipedia.
Decide using ONLY the evidence, not your own knowledge.

Answer with exactly one of these labels and nothing else:
SUPPORTS - the evidence confirms the claim
REFUTES - the evidence contradicts the claim
NOT ENOUGH INFO - the evidence does not say enough to confirm or contradict the claim"""


def build_message(claim, evidence):
    if not evidence:
        lines = "(no evidence found)"
    else:
        lines = "\n".join(f"[{i}] ({ev['title']}) {ev['sentence']}" for i, ev in enumerate(evidence, 1))
    return f"Claim: {claim}\n\nEvidence:\n{lines}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--rerank", action="store_true", help="add the cross-encoder reranker")
    args = parser.parse_args()

    out_file = ROOT / "results" / ("v2_rerank_predictions.jsonl" if args.rerank else "v2_predictions.jsonl")
    claims = [json.loads(line) for line in open(TEST_FILE, encoding="utf-8")]
    if args.limit:
        claims = claims[: args.limit]

    done = set()
    if out_file.exists():
        done = {json.loads(line)["id"] for line in open(out_file, encoding="utf-8")}
    todo = [c for c in claims if c["id"] not in done]
    print(f"Model: {MODEL}. Reranker: {'on' if args.rerank else 'off'}. {len(done)} already done, {len(todo)} to go.")

    retriever = Retriever(rerank=args.rerank)
    with open(out_file, "a", encoding="utf-8") as out:
        for n, c in enumerate(tqdm(todo), 1):
            evidence = retriever.get_evidence(c["claim"])
            raw = ask(SYSTEM_PROMPT, build_message(c["claim"], evidence))
            # The evidence is saved too, so we can later check why an answer was right or wrong.
            out.write(json.dumps({"id": c["id"], "pred": parse_label(raw), "raw": raw, "evidence": evidence}) + "\n")
            out.flush()
            if n % 10 == 0:
                save_cache()
            time.sleep(2)
    save_cache()
    print(f"Done. Predictions in {out_file}")


if __name__ == "__main__":
    main()
