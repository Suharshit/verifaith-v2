"""Public data types. These are the contract for the library, the API and the SDK."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class Label(str, Enum):
    SUPPORTED = "supported"  # some evidence entails the claim
    CONTRADICTED = "contradicted"  # evidence conflicts with the claim, none supports it
    UNSUPPORTED = "unsupported"  # the context says nothing that settles the claim


class Claim(BaseModel):
    id: int
    text: str
    source: Literal["llm", "sentence"] = "llm"


class Evidence(BaseModel):
    text: str
    context_index: int = Field(description="Which context document the evidence came from")
    start_sentence: int
    end_sentence: int


class NLIScores(BaseModel):
    entailment: float
    neutral: float
    contradiction: float


class ClaimVerdict(BaseModel):
    claim: Claim
    label: Label
    confidence: float = Field(ge=0.0, le=1.0)
    support_score: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Highest entailment probability over all candidate evidence. A continuous "
        "score for ranking and threshold calibration; the label compares it to support_threshold.",
    )
    evidence: Evidence | None = None
    scores: NLIScores | None = None
    conflicting_evidence: Evidence | None = Field(
        None,
        description="Set on a supported claim when other evidence contradicts it: the sources "
        "disagree, so the claim is grounded in one source but not settled.",
    )


Verdict = Literal["faithful", "partial", "unfaithful", "no_claims"]


class EvalResult(BaseModel):
    faithfulness: float = Field(ge=0.0, le=1.0, description="supported / total claims")
    contradiction_rate: float = Field(ge=0.0, le=1.0)
    verdict: Verdict
    counts: dict[str, int]
    claims: list[ClaimVerdict]
    warnings: list[str] = []
    version: str
    config: dict = Field(
        default_factory=dict, description="Thresholds/models used, for reproducibility"
    )
