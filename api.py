"""REST API for FactCheck-RAG.

Run locally:   uvicorn api:app --reload
Interactive docs: http://localhost:8000/docs
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.service import FactCheckService, to_source

state = {}


@asynccontextmanager
async def lifespan(app):
    # Models and indexes are loaded once at startup, not on every request.
    state["service"] = FactCheckService.from_env()
    yield
    state.clear()


app = FastAPI(
    title="FactCheck-RAG API",
    description="Multilingual (IT/ES) retrieval-augmented fact-checking with cited sources.",
    version="0.1.0",
    lifespan=lifespan,
)


class ClaimRequest(BaseModel):
    claim: str = Field(..., min_length=3, max_length=1000, examples=["Il governo ha tagliato le pensioni del 20%"])
    k: int = Field(5, ge=1, le=20, description="Number of fact-checks to return")


class Source(BaseModel):
    rank: int
    score: float
    id: str
    lang: str
    claim: str
    verdict: str = ""
    url: str = ""


class SearchResponse(BaseModel):
    claim: str
    sources: list[Source]


class VerifyResponse(SearchResponse):
    answer: str


@app.get("/health")
def health():
    service = state.get("service")
    if service is None:
        raise HTTPException(status_code=503, detail="Service is starting")
    return {"status": "ok", "factchecks": service.n_factchecks, **service.config}


@app.post("/search", response_model=SearchResponse)
def search(req: ClaimRequest):
    """Retrieval only: return the most similar published fact-checks (no LLM call)."""
    docs = state["service"].search(req.claim, k=req.k)
    return {"claim": req.claim, "sources": [to_source(fc, s, i) for i, (fc, s) in enumerate(docs, 1)]}


@app.post("/verify", response_model=VerifyResponse)
def verify(req: ClaimRequest):
    """Retrieval + LLM: verdict (TRUE / FALSE / NOT VERIFIABLE) with citations [1], [2]..."""
    try:
        answer, docs = state["service"].verify(req.claim, k=req.k)
    except Exception as e:  # LLM server down, wrong model name, timeout...
        raise HTTPException(status_code=502, detail=f"LLM call failed: {e}") from e
    return {
        "claim": req.claim,
        "answer": answer,
        "sources": [to_source(fc, s, i) for i, (fc, s) in enumerate(docs, 1)],
    }
