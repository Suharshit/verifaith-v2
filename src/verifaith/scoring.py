from __future__ import annotations

from verifaith.config import VerifierConfig
from verifaith.schemas import ClaimVerdict, Label, Verdict


def aggregate(
    verdicts: list[ClaimVerdict], cfg: VerifierConfig
) -> tuple[float, float, Verdict, dict]:
    counts = {label.value: 0 for label in Label}
    for v in verdicts:
        counts[v.label.value] += 1
    # Supported claims whose sources disagree. Not a label: they are still grounded in a source.
    counts["conflicting"] = sum(v.conflicting_evidence is not None for v in verdicts)
    total = len(verdicts)
    counts["total"] = total
    if total == 0:
        return 0.0, 0.0, "no_claims", counts
    faithfulness = counts["supported"] / total
    contradiction_rate = counts["contradicted"] / total
    settled = counts["contradicted"] == 0 and counts["conflicting"] == 0
    if faithfulness >= cfg.faithful_threshold and settled:
        verdict: Verdict = "faithful"
    elif faithfulness < cfg.unfaithful_threshold:
        verdict = "unfaithful"
    else:
        verdict = "partial"
    return round(faithfulness, 4), round(contradiction_rate, 4), verdict, counts
