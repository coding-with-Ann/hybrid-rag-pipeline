from collections.abc import Sequence

import pytest

from rag_pipeline import Document, RAGPipeline
from rag_pipeline.chunking import TextChunker


class FakeEmbedder:
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        batch = list(texts)
        self.calls.append(batch)
        vectors: list[list[float]] = []
        for text in batch:
            if "refund" in text.lower():
                vectors.append([1.0, 0.0])
            elif "query" in text.lower() or "policy" in text.lower():
                vectors.append([0.8, 0.2])
            else:
                vectors.append([0.0, 1.0])
        return vectors


class FakeLLM:
    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts.append(prompt)
        return "Refunds are handled by the customer policy."


def test_pipeline_ingests_retrieves_and_generates_with_mocked_boundaries() -> None:
    embedder = FakeEmbedder()
    llm = FakeLLM()
    pipeline = RAGPipeline(embedder=embedder, llm=llm, chunker=TextChunker(chunk_size=20, overlap=0))

    chunk_count = pipeline.ingest(
        [
            Document(id="policy", text="Refund policy allows returns within thirty days."),
            Document(id="ops", text="Warehouse pick lists are generated every morning."),
        ]
    )
    result = pipeline.query("refund policy query", top_k=1)

    assert chunk_count == 2
    assert result.answer == "Refunds are handled by the customer policy."
    assert result.sources[0].document_id == "policy"
    assert "Refund policy allows returns" in llm.prompts[0]
    assert embedder.calls[0] == [
        "Refund policy allows returns within thirty days.",
        "Warehouse pick lists are generated every morning.",
    ]
    assert embedder.calls[1] == ["refund policy query"]


def test_pipeline_requires_ingestion_before_query() -> None:
    pipeline = RAGPipeline(embedder=FakeEmbedder(), llm=FakeLLM())

    with pytest.raises(RuntimeError, match="ingest documents"):
        pipeline.query("anything")
