import pytest

from rag_pipeline.chunking import TextChunker
from rag_pipeline.schemas import Document


def test_chunker_splits_with_overlap_and_metadata() -> None:
    chunker = TextChunker(chunk_size=4, overlap=1)

    chunks = chunker.split([Document(id="doc", text="one two three four five six", metadata={"path": "a.md"})])

    assert [chunk.text for chunk in chunks] == ["one two three four", "four five six"]
    assert chunks[0].id == "doc:0"
    assert chunks[1].metadata == {"path": "a.md", "chunk_index": 1}


def test_chunker_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        TextChunker(chunk_size=10, overlap=10)
