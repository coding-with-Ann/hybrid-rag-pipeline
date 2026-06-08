from __future__ import annotations

import re
from collections.abc import Iterable

from rag_pipeline.schemas import Chunk, Document


class TextChunker:
    def __init__(self, chunk_size: int = 220, overlap: int = 40) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0:
            raise ValueError("overlap must be non-negative")
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, documents: Iterable[Document]) -> list[Chunk]:
        chunks: list[Chunk] = []
        for document in documents:
            words = _tokenize_words(document.text)
            if not words:
                continue

            start = 0
            ordinal = 0
            while start < len(words):
                end = min(start + self.chunk_size, len(words))
                text = " ".join(words[start:end])
                chunks.append(
                    Chunk(
                        id=f"{document.id}:{ordinal}",
                        document_id=document.id,
                        text=text,
                        metadata={**document.metadata, "chunk_index": ordinal},
                    )
                )
                if end == len(words):
                    break
                start = end - self.overlap
                ordinal += 1
        return chunks


def _tokenize_words(text: str) -> list[str]:
    return re.findall(r"\S+", text.strip())
