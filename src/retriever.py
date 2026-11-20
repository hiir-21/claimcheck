"""
The retriever: finds evidence sentences on Wikipedia for a claim.

How it works, for one claim:
1. SEARCH   - ask Wikipedia's search API for the 3 pages that best match the claim.
2. FETCH    - download the introduction of each page (FEVER's evidence comes from page introductions).
3. SPLIT    - cut the text into sentences: these are our candidate pieces of evidence.
4. EMBED    - turn the claim and every sentence into vectors (lists of numbers that capture meaning).
5. RANK     - score each sentence by how close its vector is to the claim's vector (cosine similarity).
6. RERANK   - optional: a second, more careful model re-scores the top 20 candidates.
7. RETURN   - the top 5 sentences, each with the page it came from.

Everything downloaded from Wikipedia is saved in data/wiki_cache.json, so re-runs are fast
and we don't send Wikipedia the same request twice.

Try it on one claim:
    python src/retriever.py "The Eiffel Tower is in Berlin."
    python src/retriever.py "The Eiffel Tower is in Berlin." --rerank
"""

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import requests
from sentence_transformers import CrossEncoder, SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
CACHE_FILE = ROOT / "data" / "wiki_cache.json"
API = "https://en.wikipedia.org/w/api.php"
# Wikipedia asks every program to identify itself; requests without this are often blocked.
HEADERS = {"User-Agent": "ClaimCheck/0.1 (student research project; https://github.com/hiir-21/claimcheck)"}

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ---------- Wikipedia access, with a cache ----------

_cache = {"search": {}, "pages": {}}
if CACHE_FILE.exists():
    _cache = json.loads(CACHE_FILE.read_text(encoding="utf-8"))


def save_cache():
    CACHE_FILE.parent.mkdir(exist_ok=True)
    CACHE_FILE.write_text(json.dumps(_cache), encoding="utf-8")


def _wiki_get(params, retries=3):
    params = {**params, "format": "json", "formatversion": 2}
    for attempt in range(retries):
        try:
            r = requests.get(API, params=params, headers=HEADERS, timeout=20)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            print(f"\n  Wikipedia error: {e}. Retrying...")
            time.sleep(3 * (attempt + 1))
    return {}


def search_titles(query, n=3):
    """Step 1: the n Wikipedia page titles that best match the query."""
    key = f"{n}|{query}"
    if key in _cache["search"]:
        return _cache["search"][key]
    data = _wiki_get({"action": "query", "list": "search", "srsearch": query, "srlimit": n})
    if "query" not in data:
        return []  # request failed: return nothing, but don't cache it, so a re-run tries again
    titles = [hit["title"] for hit in data["query"].get("search", [])]
    _cache["search"][key] = titles
    return titles


def get_intro(title):
    """Step 2: the plain-text introduction of a Wikipedia page."""
    if title in _cache["pages"]:
        return _cache["pages"][title]
    data = _wiki_get({"action": "query", "prop": "extracts", "exintro": 1,
                      "explaintext": 1, "redirects": 1, "titles": title})
    pages = data.get("query", {}).get("pages", [])
    if not pages:
        return ""  # request failed: don't cache it, so a re-run tries again
    text = pages[0].get("extract", "")
    _cache["pages"][title] = text
    return text


def key_names(claim):
    """Capitalised word groups in the claim, e.g. 'Lizzy Caplan', 'Party Down', 'Don Bradman'."""
    claim = re.sub(r"\([^)]*\)", " ", claim)  # drop brackets such as '(film)'
    return re.findall(r"[A-Z][\w'’\-]*(?:\s+[A-Z][\w'’\-]*)*", claim)


def find_titles(claim, n=3):
    """
    Step 1, with fallbacks. Wikipedia's search wants pages that contain all the words,
    so a long or unusual sentence can return nothing. When that happens:
      fallback 1: search each key name in the claim separately ('Lizzy Caplan', 'Party Down')
      fallback 2: search the first 6 words of the claim
    """
    titles = search_titles(claim, n)
    if titles:
        return titles
    for name in key_names(claim)[:3]:
        for t in search_titles(name, 2):
            if t not in titles:
                titles.append(t)
    if not titles:
        titles = search_titles(" ".join(claim.split()[:6]), n)
    return titles[: n + 1]


def split_sentences(text):
    """Step 3: cut text into sentences; very short fragments are dropped."""
    text = re.sub(r"\s+", " ", text).strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(])", text)
    return [p for p in parts if len(p.split()) >= 4]


# ---------- Ranking ----------

class Retriever:
    def __init__(self, n_pages=3, k=5, rerank=False):
        self.n_pages, self.k = n_pages, k
        print(f"Loading embedding model ({EMBED_MODEL})...")
        self.embedder = SentenceTransformer(EMBED_MODEL)
        self.reranker = None
        if rerank:
            print(f"Loading reranker ({RERANK_MODEL})...")
            self.reranker = CrossEncoder(RERANK_MODEL)

    def get_evidence(self, claim):
        titles = find_titles(claim, self.n_pages)
        # Each candidate keeps its page title, because sentences often say "He" or "It"
        # instead of the name: "Paris: It is the capital..." makes the meaning clear.
        candidates = [(t, s) for t in titles for s in split_sentences(get_intro(t))]
        if not candidates:
            return []
        texts = [f"{t}: {s}" for t, s in candidates]

        # Steps 4-5: embeddings + cosine similarity (vectors are normalised, so a dot product = cosine)
        claim_vec = self.embedder.encode([claim], normalize_embeddings=True)
        sent_vecs = self.embedder.encode(texts, normalize_embeddings=True, batch_size=64)
        scores = (sent_vecs @ claim_vec.T).ravel()
        order = list(np.argsort(-scores))

        # Step 6 (optional): the cross-encoder reads claim and sentence together, which is
        # more accurate but slower, so it only re-scores the top 20 from step 5.
        if self.reranker is not None:
            top = order[:20]
            rerank_scores = self.reranker.predict([(claim, texts[i]) for i in top])
            order = [top[j] for j in np.argsort(-rerank_scores)]
            scores = {top[j]: float(rerank_scores[j]) for j in range(len(top))}
            return [{"title": candidates[i][0], "sentence": candidates[i][1], "score": round(scores[i], 4)}
                    for i in order[: self.k]]

        return [{"title": candidates[i][0], "sentence": candidates[i][1], "score": round(float(scores[i]), 4)}
                for i in order[: self.k]]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("claim")
    parser.add_argument("--rerank", action="store_true")
    args = parser.parse_args()

    retriever = Retriever(rerank=args.rerank)
    print(f"\nPages found: {find_titles(args.claim)}")
    print(f"\nTop evidence for: {args.claim}\n")
    for i, ev in enumerate(retriever.get_evidence(args.claim), 1):
        print(f"[{i}] ({ev['title']}, score {ev['score']}) {ev['sentence']}\n")
    save_cache()
