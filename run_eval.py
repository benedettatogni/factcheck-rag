"""Compare retrievers. Usage:
  python run_eval.py --sample                 # toy data, BM25 only (no downloads)
  python run_eval.py --sample --dense         # + embeddings and hybrid
  python make_queries.py --xfact x-fact/data/x-fact   # once, needs an LLM
  python run_eval.py --xfact x-fact/data/x-fact --dense
"""
import argparse
import json

from src.data import Query, load_jsonl, load_multiclaim, load_xfact
from src.evaluate import evaluate
from src.retrieval import BM25Retriever, DenseRetriever, HybridRetriever

p = argparse.ArgumentParser()
p.add_argument("--sample", action="store_true")
p.add_argument("--multiclaim")
p.add_argument("--xfact", help="folder x-fact/data/x-fact")
p.add_argument("--queries", default="data/xfact_queries.jsonl")
p.add_argument("--dense", action="store_true")
p.add_argument("--k", type=int, default=10)
args = p.parse_args()

if args.xfact:
    fcs = load_xfact(args.xfact)
    qs = [Query(**{k: v for k, v in json.loads(l).items() if k != "kind"}) for l in open(args.queries, encoding="utf-8")]
elif args.multiclaim:
    fcs, qs = load_multiclaim(args.multiclaim)
else:
    fcs, qs = load_jsonl("data/sample_factchecks.jsonl", "data/sample_queries.jsonl")
print(f"{len(fcs)} fact-checks, {len(qs)} queries\n")

bm25 = BM25Retriever(fcs)
systems = {"BM25": bm25}
if args.dense:
    dense = DenseRetriever(fcs)
    systems["Dense (e5)"] = dense
    systems["Hybrid (RRF)"] = HybridRetriever([bm25, dense])

print(f"{'System':<14} {'success@' + str(args.k):>11} {'MRR':>7}")
for name, r in systems.items():
    m = evaluate(r.search, qs, args.k)
    print(f"{name:<14} {m[f'success@{args.k}']:>11.3f} {m['MRR']:>7.3f}")
