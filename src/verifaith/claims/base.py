from __future__ import annotations

from typing import Protocol


class ExtractionError(RuntimeError):
    """The claim extractor failed (upstream LLM error, timeout, unparseable output)."""


class ClaimExtractor(Protocol):
    source: str  # "llm" | "sentence"

    def extract(self, answer: str) -> list[str]: ...
