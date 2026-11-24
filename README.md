# FactCheck-RAG

Multilingual (Italian / Spanish) retrieval-augmented fact-checking assistant.
Given a claim, it retrieves previously published fact-checks, reranks them and asks an LLM
to give a verdict with cited sources.

> Status: work in progress. Retrieval, evaluation, REST API and Docker setup are in place; full X-Fact results are next.

## Architecture

```
Fact-checks ES/IT -> cleaning -> BM25 + multilingual embeddings -> FAISS index
Claim -> hybrid retrieval (RRF) -> cross-encoder reranker -> LLM with citations -> REST API (FastAPI) / Streamlit UI
```

The pipeline lives in `src/service.py`; `api.py` (FastAPI) and `app.py` (Streamlit) are thin layers on top of it.

## Quick start

```bash
pip install -r requirements.txt
python run_eval.py --sample            # BM25 on toy data, no downloads
python run_eval.py --sample --dense    # + multilingual-e5 and hybrid retrieval
```

For the demo, start an LLM (e.g. `ollama pull llama3.1`) and run:

```bash
export LLM_BASE_URL=http://localhost:11434/v1 LLM_MODEL=llama3.1
streamlit run app.py
```

## REST API

```bash
uvicorn api:app --reload        # interactive docs at http://localhost:8000/docs
```

| Method | Endpoint | What it does |
| --- | --- | --- |
| GET | `/health` | Status, number of fact-checks loaded, active components |
| POST | `/search` | Retrieval only: the most similar fact-checks, no LLM call |
| POST | `/verify` | Retrieval + LLM: verdict with cited sources |

```bash
curl -X POST http://localhost:8000/verify \
  -H "Content-Type: application/json" \
  -d '{"claim": "Il vaccino MPR causa l\'autismo", "k": 3}'
```

Example response (shortened):

```json
{
  "claim": "Il vaccino MPR causa l'autismo",
  "answer": "FALSE. Lo studio che collegava il vaccino MPR all'autismo è stato ritirato [1].",
  "sources": [{"rank": 1, "score": 0.97, "id": "fc6", "lang": "ita", "claim": "Il vaccino MPR non causa l'autismo: ...", "verdict": "FALSE", "url": ""}]
}
```

Configuration through environment variables: `DATA_PATH` or `XFACT_DIR` (knowledge base),
`USE_DENSE` and `USE_RERANKER` (`1`/`0`), `LLM_BASE_URL`, `LLM_MODEL`, `LLM_API_KEY`.
With `USE_DENSE=0 USE_RERANKER=0` the API starts in seconds with BM25 only, without downloading models.

## Docker

```bash
docker compose up --build
docker compose exec ollama ollama pull llama3.1   # first run only
```

This starts the API on port 8000 and a local LLM with Ollama. Hugging Face models are cached in a
volume, so they are downloaded only once. The image uses CPU-only PyTorch to stay small.

## Tests

```bash
pytest -q        # BM25 only, LLM mocked: runs in about a second, no downloads
```

## Data

- `data/` contains a small **toy sample written by hand** for testing the code; it is not real fact-checking data.
- The knowledge base is [X-Fact](https://github.com/utahnlp/x-fact) (Gupta & Srikumar, ACL 2021):
  2,912 claims in Spanish (Chequeado) and Italian (Pagella Politica, AGI) with professional verdicts.
- Test queries are **synthetic**: an LLM paraphrases each sampled claim and translates it IT<->ES
  (`make_queries.py`); a sample is checked by hand.

```bash
git clone https://github.com/utahnlp/x-fact
python make_queries.py --xfact x-fact/data/x-fact --n 200
python run_eval.py --xfact x-fact/data/x-fact --dense
```

## Results

### Sample data (sanity check)

10 fact-checks and 6 test queries (3 Italian, 3 Spanish, two of them cross-lingual), from `data/`.
Reproduce with `python run_eval.py --sample --dense` (add `--k 1` for success@1).

| Retriever | success@1 | MRR (k=10) |
| --- | --- | --- |
| BM25 | 0.833 | 0.857 |
| Dense (multilingual-e5-base) | **1.000** | **1.000** |
| Hybrid (RRF) | 0.833 | 0.875 |

BM25 misses one cross-lingual query: the Spanish claim *"Las estelas de los aviones son químicos para controlarnos"* only reaches the Italian fact-check on chemtrails (*"Le scie chimiche non esistono..."*) at rank 7, because the two texts share no words. Multilingual dense retrieval ranks it first; in the hybrid, the low BM25 rank pulls it down to rank 4.

These numbers come from a toy set of six queries and are only a sanity check: success@10 is trivially 1.0 with ten documents, and no conclusion about hybrid vs dense retrieval should be drawn from them.

### X-Fact (in progress)

Synthetic paraphrased and cross-lingual queries generated with `make_queries.py`; results will be reported separately for same-language and cross-lingual queries.

| Retriever | success@10 | MRR |
| --- | --- | --- |
| BM25 | – | – |
| Dense (multilingual-e5) | – | – |
| Hybrid (RRF) | – | – |
| Hybrid + reranker | – | – |
| Hybrid + reranker | – | – |

## Limitations

- BM25 cannot match a claim to a fact-check written in another language; dense retrieval is needed for that.
- The system can only verify claims that have already been fact-checked.
