"""
V2: the LLM answers using evidence retrieved from Wikipedia (RAG).

Same test set, same LLM and same grader as V1. The only change: before asking the LLM,
we retrieve 5 evidence sentences and tell it to decide from those, not from memory.

Run:  python src/rag_llm.py --limit 10            (quick test)
      python src/rag_llm.py                       (V2: embeddings only)
      python src/rag_llm.py --rerank              (V2 + reranker, saved to a separate file)
      python src/rag_llm.py --hybrid              (V2.1: improved page search)
      python src/rag_llm.py --hybrid --rerank     (V2.1 + reranker)
Then: python src/evaluate.py results/v2_predictions.jsonl --name V2
      python src/evaluate.py results/v2_rerank_predictions.jsonl --name V2+rerank
"""

import argparse
import json
import time
from pathlib import Path

from tqdm import tqdm

from llm import MODEL, DailyLimitReached, ask, parse_label
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
    parser.add_argument("--hybrid", action="store_true", help="always search key names too (V2.1)")
    args = parser.parse_args()

    name = "v2" + ("_hybrid" if args.hybrid else "") + ("_rerank" if args.rerank else "")
    out_file = ROOT / "results" / f"{name}_predictions.jsonl"
    claims = [json.loads(line) for line in open(TEST_FILE, encoding="utf-8")]
    if args.limit:
        claims = claims[: args.limit]

    done = set()
    if out_file.exists():
        done = {json.loads(line)["id"] for line in open(out_file, encoding="utf-8")}
    todo = [c for c in claims if c["id"] not in done]
    print(f"Model: {MODEL}. Search: {'hybrid' if args.hybrid else 'claim'}. "
          f"Reranker: {'on' if args.rerank else 'off'}. Output: {out_file.name}. "
          f"{len(done)} already done, {len(todo)} to go.")

    retriever = Retriever(rerank=args.rerank, hybrid=args.hybrid)
    with open(out_file, "a", encoding="utf-8") as out:
        for n, c in enumerate(tqdm(todo), 1):
            evidence = retriever.get_evidence(c["claim"])
            try:
                raw = ask(SYSTEM_PROMPT, build_message(c["claim"], evidence))
            except DailyLimitReached:
                save_cache()
                print(f"\nDaily API limit reached after {n - 1} new claims. Progress is saved.")
                print("Run the same command again tomorrow to continue from here.")
                return
            # The evidence is saved too, so we can later check why an answer was right or wrong.
            out.write(json.dumps({"id": c["id"], "pred": parse_label(raw), "raw": raw,
                                  "pages": retriever.last_pages, "evidence": evidence}) + "\n")
            out.flush()
            if n % 10 == 0:
                save_cache()
            time.sleep(2)
    save_cache()
    print(f"Done. Predictions in {out_file}")


if __name__ == "__main__":
    main()
