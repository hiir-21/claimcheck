# ClaimCheck

An evidence-based AI fact-checking agent, evaluated on the FEVER benchmark.

## Results

| Version | What it adds | Accuracy | Macro-F1 |
| --- | --- | --- | --- |
| V1 | LLM only, no evidence | 0.630 | 0.585 |
| V2 | + Wikipedia retrieval (RAG) | | |
| V3 | + fine-tuned DeBERTa checker | | |
| V4 | + agent (claim splitting, re-search) | | |

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then paste your Groq API key into .env
```

## Run

```bash
python src/evaluate.py --self-test          # check the grader
python src/prepare_data.py                  # build the 300-claim FEVER test set
python src/baseline_llm.py --limit 10       # quick V1 test
python src/baseline_llm.py                  # full V1 run
python src/evaluate.py results/v1_predictions.jsonl --name V1
```
