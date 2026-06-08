from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence

from rag_pipeline.schemas import Chunk, Source


class HybridRetriever:
    def __init__(
        self,
        vector_weight: float = 0.65,
        bm25_weight: float = 0.35,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if vector_weight < 0 or bm25_weight < 0:
            raise ValueError("retrieval weights must be non-negative")
        if vector_weight == 0 and bm25_weight == 0:
            raise ValueError("at least one retrieval weight must be positive")
        self.vector_weight = vector_weight
        self.bm25_weight = bm25_weight
        self.k1 = k1
        self.b = b

    def retrieve(
        self,
        query: str,
        query_embedding: Sequence[float],
        chunks: Sequence[Chunk],
        chunk_embeddings: Sequence[Sequence[float]],
        top_k: int = 4,
    ) -> list[Source]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if len(chunks) != len(chunk_embeddings):
            raise ValueError("chunks and chunk_embeddings must have matching lengths")
        if not chunks:
            return []

        vector_scores = [_cosine(query_embedding, embedding) for embedding in chunk_embeddings]
        bm25_scores = _bm25_scores(query, chunks, self.k1, self.b)
        fused_scores = _fuse(
            vector_scores,
            bm25_scores,
            vector_weight=self.vector_weight,
            bm25_weight=self.bm25_weight,
        )

        ranked = sorted(
            zip(chunks, fused_scores, strict=True),
            key=lambda item: (item[1], item[0].id),
            reverse=True,
        )
        return [
            Source(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                text=chunk.text,
                score=score,
                metadata=chunk.metadata,
            )
            for chunk, score in ranked[:top_k]
        ]


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions must match")
    numerator = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _bm25_scores(query: str, chunks: Sequence[Chunk], k1: float, b: float) -> list[float]:
    corpus_tokens = [_tokens(chunk.text) for chunk in chunks]
    query_tokens = _tokens(query)
    if not query_tokens:
        return [0.0] * len(chunks)

    document_count = len(corpus_tokens)
    avg_len = sum(len(tokens) for tokens in corpus_tokens) / max(document_count, 1)
    doc_freq: Counter[str] = Counter()
    for tokens in corpus_tokens:
        doc_freq.update(set(tokens))

    scores: list[float] = []
    for tokens in corpus_tokens:
        term_freq = Counter(tokens)
        doc_len = len(tokens)
        score = 0.0
        for term in query_tokens:
            if term_freq[term] == 0:
                continue
            idf = math.log(1 + (document_count - doc_freq[term] + 0.5) / (doc_freq[term] + 0.5))
            denominator = term_freq[term] + k1 * (1 - b + b * doc_len / max(avg_len, 1.0))
            score += idf * (term_freq[term] * (k1 + 1)) / denominator
        scores.append(score)
    return scores


def _fuse(
    vector_scores: Sequence[float],
    bm25_scores: Sequence[float],
    vector_weight: float,
    bm25_weight: float,
) -> list[float]:
    normalized_vectors = _minmax(vector_scores)
    normalized_bm25 = _minmax(bm25_scores)
    return [
        vector_weight * vector + bm25_weight * bm25
        for vector, bm25 in zip(normalized_vectors, normalized_bm25, strict=True)
    ]


def _minmax(scores: Sequence[float]) -> list[float]:
    if not scores:
        return []
    low = min(scores)
    high = max(scores)
    if high == low:
        return [0.0] * len(scores)
    return [(score - low) / (high - low) for score in scores]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())
