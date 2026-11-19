"""Tests for the pipeline, run without downloading models (BM25 only, LLM mocked).

    pytest -q
"""
import src.service as service_module
from src.data import load_jsonl
from src.service import FactCheckService, to_source


def make_service():
    factchecks, _ = load_jsonl("data/sample_factchecks.jsonl")
    return FactCheckService(factchecks, use_dense=False, use_reranker=False)


def test_search_returns_k_results():
    svc = make_service()
    docs = svc.search("vaccini", k=3)
    assert 0 < len(docs) <= 3
    scores = [s for _, s in docs]
    assert scores == sorted(scores, reverse=True)


def test_empty_claim_returns_nothing():
    assert make_service().search("   ") == []


def test_verify_calls_llm_with_retrieved_docs(monkeypatch):
    seen = {}

    def fake_generate(claim, docs):
        seen["n_docs"] = len(docs)
        return "FALSE [1]"

    monkeypatch.setattr(service_module, "generate", fake_generate)
    answer, docs = make_service().verify("vaccini", k=2)
    assert answer == "FALSE [1]"
    assert seen["n_docs"] == len(docs) > 0


def test_to_source_is_json_ready():
    fc, score = make_service().search("vaccini", k=1)[0]
    src = to_source(fc, score, 1)
    assert src["rank"] == 1 and {"id", "lang", "claim", "verdict", "url", "score"} <= src.keys()
