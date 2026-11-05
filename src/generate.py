"""Answer generation with citations.

Uses any OpenAI-compatible API. For a free local model, install Ollama and run e.g.
`ollama pull llama3.1`, then set LLM_BASE_URL=http://localhost:11434/v1 and LLM_MODEL=llama3.1.
"""
import os

SYSTEM = (
    "You are a fact-checking assistant. Answer ONLY using the numbered fact-checks provided. "
    "Reply in the same language as the user's claim. Start with one verdict: "
    "TRUE, FALSE, or NOT VERIFIABLE (if no fact-check is about this claim). "
    "Then explain in 2-4 sentences and cite sources as [1], [2]. Never invent facts or sources."
)


def build_prompt(claim, docs):
    context = "\n".join(f"[{i}] ({fc.lang}) {fc.claim} | verdict: {fc.verdict or 'n/a'} | {fc.url}"
                        for i, (fc, _) in enumerate(docs, 1))
    return f"Fact-checks:\n{context}\n\nClaim to verify: {claim}"


def generate(claim, docs):
    from openai import OpenAI

    client = OpenAI(base_url=os.getenv("LLM_BASE_URL"), api_key=os.getenv("LLM_API_KEY", "ollama"))
    resp = client.chat.completions.create(
        model=os.getenv("LLM_MODEL", "llama3.1"),
        temperature=0,
        messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": build_prompt(claim, docs)}],
    )
    return resp.choices[0].message.content
