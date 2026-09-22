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
_HEADING = re.compile(r"^\s*#{1,6}\s+")
_LIST_ITEM = re.compile(r"^\s*(?:[-*+•]|\d{1,3}[.)])\s+")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP = re.compile(r"^:?-+:?$")

# The NLI model reads at most 512 tokens and silently drops the rest of the evidence, so no
# sentence or window may come close to that. ~1.3 tokens per word leaves room for the claim.
MAX_SENTENCE_WORDS = 120
MAX_WINDOW_WORDS = 250


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _cells(row: str) -> list[str]:
    return [c.strip() for c in row.strip().strip("|").split("|")]


def _table_sentences(rows: list[str]) -> list[str]:
    """Markdown table -> one sentence per row, labelled with the column names.

    A bare row like "| Pro | $25 | 5 |" entails nothing; "Plan: Pro; Price: $25; Seats: 5." does.
    """
    body = [_cells(r) for r in rows if not all(_TABLE_SEP.match(c) for c in _cells(r) if c)]
    if len(body) < 2:
        return [" ".join(c for c in cells if c) for cells in body]
    header, out = body[0], []
    for cells in body[1:]:
        parts = [f"{h}: {v}" if h else v for h, v in zip(header, cells, strict=False) if v]
        if parts:
            out.append("; ".join(parts) + ".")
    return out


def _blocks(text: str) -> list[tuple[str, bool]]:
    """Split on line structure: blank lines, headings, list items and tables are boundaries.

    Returns (text, atomic) pairs; atomic blocks are already single sentences (table rows).
    Two lines are only joined when the line break is mid-sentence (a hard wrap): the first line
    has no closing punctuation and the next starts in lowercase.
    """
    lines = text.splitlines()
    blocks: list[tuple[str, bool]] = []
    para: list[str] = []

    def flush() -> None:
        if para:
            blocks.append((" ".join(para), False))
            para.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            flush()
        elif _TABLE_ROW.match(line):
            flush()
            j = i
            while j < len(lines) and _TABLE_ROW.match(lines[j]):
                j += 1
            blocks.extend((s, True) for s in _table_sentences(lines[i:j]))
            i = j
            continue
        elif _HEADING.match(line) or _LIST_ITEM.match(line):
            flush()
            para.append(_LIST_ITEM.sub("", _HEADING.sub("", line)).strip())
        elif (
            para
            and not para[-1].endswith((".", "!", "?", ":", ";"))
            and line.lstrip()[:1].islower()
        ):
            para.append(line.strip())
        else:
            flush()
            para.append(line.strip())
        i += 1
    flush()
    return blocks


def _split_prose(text: str) -> list[str]:
    parts = _BOUNDARY.split(normalize(text))
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


def _chunk(sentence: str) -> list[str]:
    words = sentence.split()
    step = MAX_SENTENCE_WORDS
    return [" ".join(words[i : i + step]) for i in range(0, len(words), step)]


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []
    for block, atomic in _blocks(text):
        for s in [normalize(block)] if atomic else _split_prose(block):
            if s:
                sentences.extend(_chunk(s))
    return sentences


def make_windows(
    contexts: list[str], size: int = 3, max_words: int = MAX_WINDOW_WORDS
) -> list[Evidence]:
    """Overlapping windows of up to `size` sentences that END at each sentence.

    Every sentence appears together with the sentences before it, so a sentence like
    "It stands 330 metres tall." is checked alongside the sentence naming "It". Windows drop their
    earliest sentences to stay within `max_words`, so the NLI model never truncates them.
    """
    windows: list[Evidence] = []
    for ci, ctx in enumerate(contexts):
        sents = split_sentences(ctx)
        lengths = [len(s.split()) for s in sents]
        for end in range(len(sents)):
            start = max(0, end - size + 1)
            while start < end and sum(lengths[start : end + 1]) > max_words:
                start += 1
            windows.append(
                Evidence(
                    text=" ".join(sents[start : end + 1]),
                    context_index=ci,
                    start_sentence=start,
                    end_sentence=end,
                )
            )
    return windows


def make_views(contexts: list[str], size: int = 3) -> list[list[Evidence]]:
    """Single sentences plus `size`-sentence windows.

    Windows resolve pronouns, but unrelated sentences in a window can drown out the entailment
    (0.996 -> 0.008 on DeBERTa NLI), so each sentence is also checked on its own.
    """
    views = [make_windows(contexts, size=1)]
    if size > 1:
        views.append(make_windows(contexts, size=size))
    return views
