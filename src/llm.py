"""
Shared LLM helper, used by V2 and every later version.

Keeping the LLM call in one place means every version talks to the model in exactly
the same way (same model, same settings), so the only thing that changes between
versions is what we add: evidence, a checker, an agent.
"""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = os.environ.get("GROQ_MODEL", "openai/gpt-oss-120b")


def ask(system_prompt, user_message, retries=5):
    """Send one prompt to the LLM and return its text reply. Retries on rate limits."""
    request = dict(
        model=MODEL,
        temperature=0,
        max_tokens=1000,
        reasoning_effort="low",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(**request)
            return (response.choices[0].message.content or "").strip()
        except TypeError:
            request.pop("reasoning_effort", None)
            continue
        except Exception as e:
            wait = 10 * (attempt + 1)
            print(f"\n  LLM error: {e}. Waiting {wait}s and retrying...")
            time.sleep(wait)
    return "ERROR"


def parse_label(text):
    """Turn the model's reply into one of the three labels."""
    t = text.upper()
    if "NOT ENOUGH" in t:
        return "NOT ENOUGH INFO"
    if "REFUTE" in t:
        return "REFUTES"
    if "SUPPORT" in t:
        return "SUPPORTS"
    return "NOT ENOUGH INFO"
