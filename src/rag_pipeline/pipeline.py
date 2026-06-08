from __future__ import annotations

from collections.abc import Iterable

from rag_pipeline.chunking import TextChunker
from rag_pipeline.embeddings import Embedder
from rag_pipeline.generation import LLM, PromptBuilder
from rag_pipeline.retrieval import HybridRetriever
from rag_pipeline.schemas import Chunk, Document, QueryResult


class RAGPipeline:
    def __init__(
        self,
        embedder: Embedder,
        llm: LLM,
        chunker: TextChunker | None = None,
        retriever: HybridRetriever | None = None,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        self._embedder = embedder
        self._llm = llm
        self._chunker = chunker or TextChunker()
        self._retriever = retriever or HybridRetriever()
        self._prompt_builder = prompt_builder or PromptBuilder()
        self._chunks: list[Chunk] = []
        self._chunk_embeddings: list[list[float]] = []

    def ingest(self, documents: Iterable[Document]) -> int:
        chunks = self._chunker.split(documents)
        embeddings = self._embedder.embed([chunk.text for chunk in chunks])
        if len(embeddings) != len(chunks):
            raise ValueError("embedder returned a different number of embeddings than inputs")

        self._chunks = chunks
        self._chunk_embeddings = embeddings
        return len(chunks)

    def query(self, question: str, top_k: int = 4) -> QueryResult:
        if not self._chunks:
            raise RuntimeError("ingest documents before querying")

        query_embedding = self._embedder.embed([question])[0]
        sources = self._retriever.retrieve(
            question,
            query_embedding,
            self._chunks,
            self._chunk_embeddings,
            top_k=top_k,
        )
        prompt = self._prompt_builder.build(question, sources)
        answer = self._llm.complete(prompt).strip()
        return QueryResult(answer=answer, sources=sources)
