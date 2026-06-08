# Hybrid RAG Pipeline

Production-ready Python skeleton for a retrieval-augmented generation pipeline using LlamaIndex concepts:

- deterministic document chunking
- explicit embedding and LLM boundaries
- hybrid BM25/vector retrieval
- configurable score fusion
- source-grounded answer generation
- PyTest coverage with mocked embeddings and LLM calls

The package is intentionally small and dependency-light at runtime boundaries. The core pipeline accepts protocols for embedders and LLMs, so tests and local development do not require network access.

## Layout

```text
src/rag_pipeline/
  chunking.py       document chunking
  embeddings.py     embedding protocol and deterministic local embedder
  generation.py     LLM protocol and prompt construction
  pipeline.py       ingestion and query orchestration
  retrieval.py      hybrid BM25/vector retriever
  schemas.py        typed data models
tests/
  test_*.py
```

## Install

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Run Tests

```bash
pytest
```

## Example

```python
from rag_pipeline import Document, RAGPipeline
from rag_pipeline.embeddings import HashingEmbedder
from rag_pipeline.generation import EchoLLM

pipeline = RAGPipeline(embedder=HashingEmbedder(dimensions=128), llm=EchoLLM())
pipeline.ingest([
    Document(id="doc-1", text="Hybrid search combines sparse lexical matching with dense vectors.")
])

answer = pipeline.query("What does hybrid search combine?")
print(answer.answer)
print(answer.sources)
```

## Production Integration

Implement `Embedder` and `LLM` protocols with your production services. A LlamaIndex-backed adapter should live at the boundary, for example:

```python
class LlamaIndexEmbedder:
    def __init__(self, model):
        self._model = model

    def embed(self, texts):
        return self._model.get_text_embedding_batch(texts)
```

Keep network clients, credentials, retries, and telemetry in adapters rather than inside the retrieval core. This keeps tests deterministic and makes production concerns explicit.
