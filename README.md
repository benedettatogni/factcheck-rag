# FactCheck-RAG

Multilingual (Italian / Spanish) retrieval-augmented fact-checking assistant.
Given a claim, it retrieves previously published fact-checks, reranks them and asks an LLM
to give a verdict with cited sources.

> Status: work in progress (week 1–2 of 5).

## Architecture

```
Fact-checks ES/IT -> cleaning -> BM25 + multilingual embeddings -> FAISS index
Claim -> hybrid retrieval (RRF) -> cross-encoder reranker -> LLM with citations -> Streamlit UI
```

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

| Retriever | success@10 | MRR |
| --- | --- | --- |
| BM25 | – | – |
| Dense (multilingual-e5) | – | – |
| Hybrid (RRF) | – | – |
| Hybrid + reranker | – | – |

## Limitations

- BM25 cannot match a claim to a fact-check written in another language; dense retrieval is needed for that.
- The system can only verify claims that have already been fact-checked.
