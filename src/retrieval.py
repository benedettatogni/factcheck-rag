"""Retrievers: BM25 baseline, dense (multilingual embeddings + FAISS), hybrid (RRF) and reranker."""
import math
import re
from collections import Counter

import numpy as np


def tokenize(text):
    return re.findall(r"\w+", text.lower())


class BM25Retriever:
    """Okapi BM25, written out so every step is visible (k1 and b are the standard defaults)."""

    def __init__(self, factchecks, k1=1.5, b=0.75):
        self.fcs, self.k1, self.b = factchecks, k1, b
        self.docs = [Counter(tokenize(fc.claim)) for fc in factchecks]
        self.lengths = np.array([sum(d.values()) for d in self.docs], dtype=float)
        self.avg_len = self.lengths.mean() if len(self.docs) else 0.0
        df = Counter(term for d in self.docs for term in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query):
        out = np.zeros(len(self.docs))
        norm = self.k1 * (1 - self.b + self.b * self.lengths / self.avg_len)
        for term in set(tokenize(query)):
            if term not in self.idf:
                continue
            tf = np.array([d.get(term, 0) for d in self.docs], dtype=float)
            out += self.idf[term] * tf * (self.k1 + 1) / (tf + norm)
        return out

    def search(self, query, k=10):
        scores = self.scores(query)
        top = np.argsort(scores)[::-1][:k]
        return [(self.fcs[i], float(scores[i])) for i in top]


class DenseRetriever:
    """Multilingual embeddings. E5 models need the 'query: ' / 'passage: ' prefixes."""

    def __init__(self, factchecks, model_name="intfloat/multilingual-e5-base"):
        import faiss
        from sentence_transformers import SentenceTransformer

        self.fcs = factchecks
        self.model = SentenceTransformer(model_name)
        self.is_e5 = "e5" in model_name.lower()
        passages = [("passage: " if self.is_e5 else "") + fc.claim for fc in factchecks]
        emb = self.model.encode(passages, normalize_embeddings=True, batch_size=64, show_progress_bar=True)
        self.index = faiss.IndexFlatIP(emb.shape[1])  # inner product = cosine on normalized vectors
        self.index.add(np.asarray(emb, dtype="float32"))

    def search(self, query, k=10):
        q = self.model.encode([("query: " if self.is_e5 else "") + query], normalize_embeddings=True)
        scores, idx = self.index.search(np.asarray(q, dtype="float32"), k)
        return [(self.fcs[i], float(s)) for i, s in zip(idx[0], scores[0]) if i >= 0]


class HybridRetriever:
    """Combines rankings with Reciprocal Rank Fusion: score = sum 1 / (rrf_k + rank)."""

    def __init__(self, retrievers, rrf_k=60, depth=50):
        self.retrievers, self.rrf_k, self.depth = retrievers, rrf_k, depth

    def search(self, query, k=10):
        scores, by_id = {}, {}
        for r in self.retrievers:
            for rank, (fc, _) in enumerate(r.search(query, self.depth)):
                scores[fc.id] = scores.get(fc.id, 0.0) + 1.0 / (self.rrf_k + rank + 1)
                by_id[fc.id] = fc
        top = sorted(scores, key=scores.get, reverse=True)[:k]
        return [(by_id[i], scores[i]) for i in top]


class Reranker:
    def __init__(self, model_name="BAAI/bge-reranker-v2-m3"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, query, candidates, k=5):
        if not candidates:
            return []
        scores = self.model.predict([(query, fc.claim) for fc, _ in candidates])
        order = np.argsort(scores)[::-1][:k]
        return [(candidates[i][0], float(scores[i])) for i in order]
