"""Create synthetic test queries from X-Fact: for each sampled fact-check, an LLM writes
(1) a paraphrase in the same language and (2) a translation into the other language (IT<->ES).
The correct answer for each query is the original fact-check. Check a sample by hand!

Usage: python make_queries.py --xfact path/to/x-fact/data/x-fact --n 200
Needs LLM_BASE_URL / LLM_MODEL (see src/generate.py).
"""
import argparse
import json
import os
import random

from openai import OpenAI

from src.data import load_xfact

OTHER = {"es": "Italian", "it": "Spanish"}
SAME = {"es": "Spanish", "it": "Italian"}

p = argparse.ArgumentParser()
p.add_argument("--xfact", required=True)
p.add_argument("--n", type=int, default=200)
p.add_argument("--out", default="data/xfact_queries.jsonl")
args = p.parse_args()

client = OpenAI(base_url=os.getenv("LLM_BASE_URL"), api_key=os.getenv("LLM_API_KEY", "ollama"))
model = os.getenv("LLM_MODEL", "llama3.1")


def ask(prompt):
    r = client.chat.completions.create(model=model, temperature=0.7,
                                       messages=[{"role": "user", "content": prompt}])
    return r.choices[0].message.content.strip().strip('"')


fcs = load_xfact(args.xfact)
random.seed(42)
sample = random.sample(fcs, min(args.n, len(fcs)))
with open(args.out, "w", encoding="utf-8") as f:
    for fc in sample:
        para = ask(f"Rewrite this claim as a social media user would, in {SAME[fc.lang]}, "
                   f"keeping the same meaning. Reply with the rewritten claim only.\n\n{fc.claim}")
        trans = ask(f"Translate this claim into {OTHER[fc.lang]}. Reply with the translation only.\n\n{fc.claim}")
        for text, kind, lang in [(para, "paraphrase", fc.lang), (trans, "translation", "it" if fc.lang == "es" else "es")]:
            f.write(json.dumps({"query": text, "lang": lang, "relevant_ids": [fc.id], "kind": kind},
                               ensure_ascii=False) + "\n")
print(f"Wrote {2 * len(sample)} queries to {args.out}")
