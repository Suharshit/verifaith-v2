"""Checks that extracted claims are really IN the answer (catches the LLM 'correcting' facts)."""

from __future__ import annotations

import re

from verifaith.nli.base import EntailmentModel

_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "").rstrip(".") for n in _NUM.findall(text)}


def number_mismatches(answer: str, claims: list[str]) -> list[int]:
    """Indices of claims containing a number that never appears in the answer."""
    in_answer = _numbers(answer)
    return [i for i, c in enumerate(claims) if not _numbers(c) <= in_answer]


def unfaithful_extractions(
    answer: str, claims: list[str], nli: EntailmentModel, threshold: float
) -> list[int]:
    """Indices of claims that fail the numeric check or are not entailed by the answer."""
    bad = set(number_mismatches(answer, claims))
    if claims:
        scores = nli.predict([(answer, c) for c in claims])
        bad |= {i for i, s in enumerate(scores) if s.entailment < threshold}
    return sorted(bad)
