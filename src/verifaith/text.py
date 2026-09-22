"""Sentence splitting and evidence windows. Pure Python, no downloads needed."""

from __future__ import annotations

import re

from verifaith.schemas import Evidence

_ABBREV = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "st",
    "vs",
    "etc",
    "e.g",
    "i.e",
    "u.s",
    "u.k",
    "inc",
    "ltd",
    "co",
    "no",
    "fig",
    "approx",
    "jan",
    "feb",
    "mar",
    "apr",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
}
_BOUNDARY = re.compile(r"(?<=[.!?])[\"')\]]*\s+(?=[\"'(\[]?[A-Z0-9])")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    text = normalize(text)
    if not text:
        return []
    parts = _BOUNDARY.split(text)
    sentences: list[str] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if sentences:
            last_word = sentences[-1].rsplit(" ", 1)[-1].rstrip(".").lower()
            # Merge when the previous "sentence" ended in an abbreviation or single initial.
            if last_word in _ABBREV or (len(last_word) == 1 and last_word.isalpha()):
                sentences[-1] = f"{sentences[-1]} {part}"
                continue
        sentences.append(part)
    return sentences


def make_windows(contexts: list[str], size: int = 3) -> list[Evidence]:
    """Overlapping windows of up to `size` sentences that END at each sentence.

    Every sentence appears together with the sentences before it, so a sentence like
    "It stands 330 metres tall." is checked alongside the sentence naming "It".
    """
    windows: list[Evidence] = []
    for ci, ctx in enumerate(contexts):
        sents = split_sentences(ctx)
        for end in range(len(sents)):
            start = max(0, end - size + 1)
            windows.append(
                Evidence(
                    text=" ".join(sents[start : end + 1]),
                    context_index=ci,
                    start_sentence=start,
                    end_sentence=end,
                )
            )
    return windows
