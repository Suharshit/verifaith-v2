from __future__ import annotations

from typing import Protocol

from verifaith.schemas import NLIScores


class EntailmentModel(Protocol):
    """Scores (premise, hypothesis) pairs. Must return one NLIScores per pair, in order."""

    name: str

    def predict(self, pairs: list[tuple[str, str]]) -> list[NLIScores]: ...
