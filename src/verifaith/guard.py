"""Checks that extracted claims are really IN the answer (catches the LLM 'correcting' facts)."""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

from verifaith.nli.base import EntailmentModel
from verifaith.retrieval import Retriever, select_from_views
from verifaith.text import make_views

_UNITS = {
    w: i
    for i, w in enumerate(
        "zero one two three four five six seven eight nine ten eleven twelve thirteen "
        "fourteen fifteen sixteen seventeen eighteen nineteen".split()
    )
}
_TENS = {
    w: 10 * i
    for i, w in enumerate("twenty thirty forty fifty sixty seventy eighty ninety".split(), 2)
}
_SCALES = {
    "hundred": 100,
    "thousand": 10**3,
    "million": 10**6,
    "billion": 10**9,
    "trillion": 10**12,
}
_WORD = "|".join([*_TENS, *_UNITS])
_NUM = re.compile(
    rf"(?:(?P<digits>\d[\d,]*(?:\.\d+)?)|\b(?P<word>(?:{_WORD})(?:-(?:{_WORD}))?)\b)"
    rf"(?:\s+(?P<scale>{'|'.join(_SCALES)})\b)?",
    re.IGNORECASE,
)


def _numbers(text: str, keep_unscaled: bool = False) -> set[Decimal]:
    """Numeric VALUES in the text, so "three" == "3" and "1.5 million" == "1,500,000".

    `keep_unscaled` also keeps 1.5 for "1.5 million": lenient for the answer, never for a claim.
    """
    out: set[Decimal] = set()
    for m in _NUM.finditer(text):
        if m["digits"]:
            try:
                value = Decimal(m["digits"].replace(",", "").rstrip("."))
            except InvalidOperation:
                continue
        else:
            value = Decimal(
                sum(_UNITS.get(p, _TENS.get(p, 0)) for p in m["word"].lower().split("-"))
            )
        if m["scale"]:
            out.add(value * _SCALES[m["scale"].lower()])
        if keep_unscaled or not m["scale"]:
            out.add(value)
    return out


def number_mismatches(answer: str, claims: list[str]) -> list[int]:
    """Indices of claims containing a number that never appears in the answer."""
    in_answer = _numbers(answer, keep_unscaled=True)
    return [i for i, c in enumerate(claims) if not _numbers(c) <= in_answer]


def unfaithful_extractions(
    answer: str,
    claims: list[str],
    nli: EntailmentModel,
    threshold: float,
    retriever: Retriever,
    window_size: int = 3,
    max_candidates: int = 8,
) -> list[int]:
    """Indices of claims that fail the numeric check or are not entailed by the answer.

    Each claim is checked against the whole answer plus the answer sentences and windows most
    relevant to it. The whole answer alone is not enough: the NLI model truncates it, so claims from
    the end of a long answer always looked unfaithful. Windows resolve pronouns; single sentences
    are needed too, because unrelated sentences in a window can drown out the entailment.
    """
    bad = set(number_mismatches(answer, claims))
    if not claims:
        return []
    views = make_views([answer], size=window_size)
    premises = [
        [answer, *(ev.text for ev in select_from_views(retriever, c, views, max_candidates))]
        for c in claims
    ]
    pairs = [(p, c) for c, ps in zip(claims, premises, strict=True) for p in ps]
    scores = nli.predict(pairs)
    cursor = 0
    for i, ps in enumerate(premises):
        chunk = scores[cursor : cursor + len(ps)]
        cursor += len(ps)
        if max(s.entailment for s in chunk) < threshold:
            bad.add(i)
    return sorted(bad)
