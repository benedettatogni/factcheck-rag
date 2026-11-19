"""Retrieval + generation pipeline, independent of the web framework.

The API (api.py) and the Streamlit demo (app.py) both use this module, so the logic
is tested once and the web layer stays thin.

Configuration via environment variables:
    DATA_PATH       JSONL file with fact-checks (default: data/sample_factchecks.jsonl)
    XFACT_DIR       if set, load X-Fact from this folder instead (x-fact/data/x-fact)
    USE_DENSE       "1" to add multilingual dense retrieval (default "1")
    USE_RERANKER    "1" to rerank with a cross-encoder (default "1")
    LLM_BASE_URL, LLM_MODEL, LLM_API_KEY   see src/generate.py
"""
import os
from dataclasses import asdict

from src.data import load_jsonl, load_xfact
from src.generate import generate
from src.retrieval import BM25Retriever, DenseRetriever, HybridRetriever, Reranker


def _flag(name, default="1"):
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes")


class FactCheckService:
    def __init__(self, factchecks, use_dense=True, use_reranker=True):
        if not factchecks:
            raise ValueError("No fact-checks loaded: check DATA_PATH or XFACT_DIR.")
        self.n_factchecks = len(factchecks)
        retrievers = [BM25Retriever(factchecks)]
        if use_dense:
            retrievers.append(DenseRetriever(factchecks))
        self.retriever = HybridRetriever(retrievers) if len(retrievers) > 1 else retrievers[0]
        self.reranker = Reranker() if use_reranker else None
        self.config = {"dense": use_dense, "reranker": use_reranker}

    @classmethod
    def from_env(cls):
        xfact_dir = os.getenv("XFACT_DIR")
        if xfact_dir:
            factchecks = load_xfact(xfact_dir)
        else:
            factchecks, _ = load_jsonl(os.getenv("DATA_PATH", "data/sample_factchecks.jsonl"))
        return cls(factchecks, use_dense=_flag("USE_DENSE"), use_reranker=_flag("USE_RERANKER"))

    def search(self, claim, k=5):
        """Return the k most relevant fact-checks as (FactCheck, score) pairs."""
        claim = claim.strip()
        if not claim:
            return []
        candidates = self.retriever.search(claim, k=max(k, 20))
        if self.reranker:
            return self.reranker.rerank(claim, candidates, k=k)
        return candidates[:k]

    def verify(self, claim, k=5):
        """Retrieve fact-checks and ask the LLM for a verdict with citations."""
        docs = self.search(claim, k=k)
        answer = generate(claim, docs) if docs else "NOT VERIFIABLE: no related fact-check was found."
        return answer, docs


def to_source(fc, score, rank):
    """Serialise a retrieved fact-check for JSON responses."""
    return {"rank": rank, "score": round(float(score), 4), **asdict(fc)}
