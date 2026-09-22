"""Deterministic fakes so you (and your users) can test integrations without models or API keys."""

from __future__ import annotations

import re

from verifaith.schemas import NLIScores

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
_TOK = re.compile(r"[a-z0-9]+")


def _content(text: str) -> set[str]:
    return {t for t in _TOK.findall(text.lower()) if t not in _STOP}


class KeywordNLI:
    """Toy NLI: entail if every content word of the hypothesis is in the premise;
    contradict if the words match but a number differs; otherwise neutral."""

    name = "keyword-fake"

    def __init__(self):
        self.calls = 0

    def predict(self, pairs):
        self.calls += 1
        out = []
        for premise, hyp in pairs:
            p, h = _content(premise), _content(hyp)
            nums = {t for t in h if t.isdigit()}
            words = h - nums
            if h and h <= p:
                out.append(NLIScores(entailment=0.95, neutral=0.04, contradiction=0.01))
            elif nums and not nums <= p and words and words <= p and any(t.isdigit() for t in p):
                out.append(NLIScores(entailment=0.02, neutral=0.08, contradiction=0.90))
            else:
                out.append(NLIScores(entailment=0.03, neutral=0.94, contradiction=0.03))
        return out


class StaticExtractor:
    """Returns pre-set claims regardless of input (simulate an LLM extractor)."""

    source = "llm"

    def __init__(self, claims: list[str]):
        self.claims = claims

    def extract(self, answer: str) -> list[str]:
        return list(self.claims)
