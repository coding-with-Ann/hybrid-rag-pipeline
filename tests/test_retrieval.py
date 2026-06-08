import pytest

from rag_pipeline.retrieval import HybridRetriever
from rag_pipeline.schemas import Chunk


def test_hybrid_retriever_combines_bm25_and_vector_scores() -> None:
    chunks = [
        Chunk(id="a", document_id="doc-a", text="alpha beta semantic search"),
        Chunk(id="b", document_id="doc-b", text="invoice payment terms"),
        Chunk(id="c", document_id="doc-c", text="alpha contract clause"),
    ]
    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
        [0.6, 0.4],
    ]
    retriever = HybridRetriever(vector_weight=0.6, bm25_weight=0.4)

    sources = retriever.retrieve("alpha semantic", [1.0, 0.0], chunks, embeddings, top_k=2)

    assert [source.chunk_id for source in sources] == ["a", "c"]
    assert sources[0].score > sources[1].score


def test_hybrid_retriever_validates_embedding_lengths() -> None:
    retriever = HybridRetriever()

    with pytest.raises(ValueError, match="matching lengths"):
        retriever.retrieve("query", [1.0], [Chunk(id="a", document_id="d", text="text")], [], top_k=1)
