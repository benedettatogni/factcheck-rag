"""Retrieval metrics: success@k (SemEval-2025 Task 7) and MRR."""


def evaluate(search_fn, queries, k=10):
    hits, rr = 0, 0.0
    for q in queries:
        ids = [fc.id for fc, _ in search_fn(q.query, k)]
        relevant = set(q.relevant_ids)
        if relevant & set(ids):
            hits += 1
        rank = next((i + 1 for i, x in enumerate(ids) if x in relevant), None)
        rr += 1.0 / rank if rank else 0.0
    n = max(len(queries), 1)
    return {f"success@{k}": hits / n, "MRR": rr / n, "n_queries": len(queries)}
