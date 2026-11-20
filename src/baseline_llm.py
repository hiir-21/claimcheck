"""
V1 baseline: ask an LLM directly, with no evidence.

For each claim in the test set, we send one prompt to the LLM (through Groq's free API)
and ask for exactly one label. Answers are saved as we go, so if the script stops
(rate limit, internet drop), running it again continues where it left off.

Run:  python src/baseline_llm.py --limit 10     (quick test on 10 claims)
      python src/baseline_llm.py                (all claims)
Then: python src/evaluate.py results/v1_predictions.jsonl --name V1
"""

import argparse
import json
import os
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
TEST_FILE = ROOT / "data" / "fever_test.jsonl"
OUT_FILE = ROOT / "results" / "v1_predictions.jsonl"

load_dotenv(ROOT / ".env")
client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")

SYSTEM_PROMPT = """You are a careful fact-checker.
You will be given a single factual claim. Using only your own knowledge, decide whether it is true.

Answer with exactly one of these labels and nothing else:
SUPPORTS - the claim is true
REFUTES - the claim is false
NOT ENOUGH INFO - you cannot tell whether it is true or false"""


def parse_label(text):
    """Turn the model's reply into one of the three labels."""
    t = text.upper()
    if "NOT ENOUGH" in t:
        return "NOT ENOUGH INFO"
    if "REFUTE" in t:
        return "REFUTES"
    if "SUPPORT" in t:
        return "SUPPORTS"
    return "NOT ENOUGH INFO"  # an unclear answer counts as "can't tell"


def ask_llm(claim, retries=5):
    """Send one claim; wait and retry if we hit the free-tier rate limit."""
    request = dict(
        model=MODEL,
        temperature=0,  # same answer every time, so results are reproducible
        # GPT OSS is a reasoning model: it "thinks" before answering, and thinking uses tokens.
        # 1000 leaves room for that; "low" effort keeps it quick.
        max_tokens=1000,
        reasoning_effort="low",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Claim: {claim}"},
        ],
    )
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(**request)
            return (response.choices[0].message.content or "").strip()
        except TypeError:
            # Older library version without reasoning_effort: drop it and try again
            request.pop("reasoning_effort", None)
            continue
        except Exception as e:  # rate limits and network errors
            wait = 10 * (attempt + 1)
            print(f"\n  Error: {e}. Waiting {wait}s and retrying...")
            time.sleep(wait)
    return "ERROR"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="only run the first N claims (for testing)")
    args = parser.parse_args()

    claims = [json.loads(line) for line in open(TEST_FILE, encoding="utf-8")]
    if args.limit:
        claims = claims[: args.limit]

    done = set()
    if OUT_FILE.exists():
        done = {json.loads(line)["id"] for line in open(OUT_FILE, encoding="utf-8")}
    todo = [c for c in claims if c["id"] not in done]
    print(f"Model: {MODEL}. {len(done)} already done, {len(todo)} to go.")

    OUT_FILE.parent.mkdir(exist_ok=True)
    with open(OUT_FILE, "a", encoding="utf-8") as out:
        for c in tqdm(todo):
            raw = ask_llm(c["claim"])
            out.write(json.dumps({"id": c["id"], "pred": parse_label(raw), "raw": raw}) + "\n")
            out.flush()
            time.sleep(2)  # stay under the free-tier requests-per-minute limit

    print(f"Done. Predictions in {OUT_FILE}")


if __name__ == "__main__":
    main()
