"""Candidate evidence selection. Lexical by default (no model, deterministic)."""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Protocol

from verifaith.schemas import Evidence

_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set(
    [
        "a",
        "an",
        "the",
        "of",
        "in",
        "on",
        "at",
        "to",
        "is",
        "was",
        "were",
        "be",
        "been",
        "are",
        "and",
        "or",
        "for",
        "by",
        "with",
        "as",
        "it",
        "its",
        "this",
        "that",
    ]
)


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


class Retriever(Protocol):
    def select(self, claim: str, windows: list[Evidence], k: int) -> list[int]: ...


class LexicalRetriever:
    """BM25-style scoring. Returns ALL windows when there are k or fewer (never pads)."""

    def select(self, claim: str, windows: list[Evidence], k: int) -> list[int]:
        n = len(windows)
        if n <= k:
            return list(range(n))
        docs = [_tokens(w.text) for w in windows]
        df = Counter(t for d in docs for t in set(d))
        avg_len = sum(len(d) for d in docs) / n or 1.0
        q = set(_tokens(claim))
        scores = []
        for i, d in enumerate(docs):
            tf = Counter(d)
            s = 0.0
            for t in q:
                if t in tf:
                    idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                    s += idf * tf[t] * 2.2 / (tf[t] + 1.2 * (0.25 + 0.75 * len(d) / avg_len))
            scores.append((s, -i))
        ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)
        return ranked[:k]
