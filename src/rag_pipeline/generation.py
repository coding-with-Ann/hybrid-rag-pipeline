from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from rag_pipeline.schemas import Source


class LLM(Protocol):
    def complete(self, prompt: str) -> str:
        """Return a completion for the supplied prompt."""


class PromptBuilder:
    def build(self, question: str, sources: Sequence[Source]) -> str:
        context = "\n\n".join(
            f"[{index}] {source.text}" for index, source in enumerate(sources, start=1)
        )
        return (
            "Answer the question using only the provided context. "
            "If the answer is not in the context, say you do not know.\n\n"
            f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"
        )


class EchoLLM:
    """Local non-network LLM useful for smoke tests and examples."""

    def complete(self, prompt: str) -> str:
        return prompt
