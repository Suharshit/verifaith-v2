"""The verification pipeline: extract -> guard -> window -> select -> NLI -> decide -> score."""

from __future__ import annotations

from verifaith.claims.base import ClaimExtractor
from verifaith.claims.sentence import SentenceExtractor
from verifaith.config import Settings, VerifierConfig
from verifaith.guard import unfaithful_extractions
from verifaith.nli.base import EntailmentModel
from verifaith.retrieval import LexicalRetriever, Retriever
from verifaith.schemas import (
    Claim,
    ClaimVerdict,
    EvalResult,
    Evidence,
    Label,
    NLIScores,
)
from verifaith.scoring import aggregate
from verifaith.text import make_windows

VERSION = "0.1.0"


class Verifier:
    def __init__(
        self,
        extractor: ClaimExtractor,
        nli: EntailmentModel,
        retriever: Retriever | None = None,
        config: VerifierConfig | None = None,
    ):
        self.extractor = extractor
        self.nli = nli
        self.retriever = retriever or LexicalRetriever()
        self.config = config or VerifierConfig()
        self._fallback = SentenceExtractor()

    @classmethod
    def from_settings(cls, settings: Settings | None = None, config: VerifierConfig | None = None):
        from verifaith.nli.hf import HFEntailmentModel

        s = settings or Settings()
        if s.extractor == "llm":
            from verifaith.claims.llm import LLMClaimExtractor

            extractor: ClaimExtractor = LLMClaimExtractor(
                s.llm_base_url, s.llm_api_key, s.llm_model, timeout_s=s.llm_timeout_s
            )
        else:
            extractor = SentenceExtractor()
        return cls(
            extractor, HFEntailmentModel(s.nli_model, batch_size=s.nli_batch_size), config=config
        )

    # ------------------------------------------------------------------ public API
    def evaluate(self, answer: str, contexts: list[str]) -> EvalResult:
        cfg = self.config
        warnings: list[str] = []

        claims = self._extract_claims(answer, warnings)
        windows = make_windows(contexts, size=cfg.window_size)
        if not windows:
            warnings.append("Context contained no usable sentences; every claim is unsupported.")

        # One batched NLI call for every (evidence, claim) pair.
        plan: list[list[int]] = [
            self.retriever.select(c.text, windows, cfg.max_candidates) for c in claims
        ]
        pairs = [(windows[w].text, c.text) for c, ws in zip(claims, plan, strict=True) for w in ws]
        scores = self.nli.predict(pairs) if pairs else []

        verdicts, cursor = [], 0
        for claim, ws in zip(claims, plan, strict=True):
            chunk = scores[cursor : cursor + len(ws)]
            cursor += len(ws)
            verdicts.append(self._decide(claim, [windows[w] for w in ws], chunk))

        faithfulness, contradiction_rate, verdict, counts = aggregate(verdicts, cfg)
        return EvalResult(
            faithfulness=faithfulness,
            contradiction_rate=contradiction_rate,
            verdict=verdict,
            counts=counts,
            claims=verdicts,
            warnings=warnings,
            version=VERSION,
            config={
                "extractor": getattr(self.extractor, "source", "custom"),
                "nli_model": getattr(self.nli, "name", "custom"),
                **cfg.model_dump(),
            },
        )

    # ------------------------------------------------------------------ internals
    def _extract_claims(self, answer: str, warnings: list[str]) -> list[Claim]:
        texts = self.extractor.extract(answer)
        source = getattr(self.extractor, "source", "llm")
        if source == "llm" and texts:
            bad = unfaithful_extractions(
                answer, texts, self.nli, self.config.extraction_guard_threshold
            )
            if bad:
                # The extractor changed or invented facts. Never verify rewritten claims:
                # fall back to the answer's own sentences.
                warnings.append(
                    f"{len(bad)} extracted claim(s) were not faithful to the answer; "
                    "fell back to sentence-level claims."
                )
                texts, source = self._fallback.extract(answer), "sentence"
        if not texts:
            warnings.append("No factual claims found in the answer.")
        return [Claim(id=i + 1, text=t, source=source) for i, t in enumerate(texts)]

    def _decide(
        self, claim: Claim, evidence: list[Evidence], scores: list[NLIScores]
    ) -> ClaimVerdict:
        cfg = self.config
        if not scores:
            return ClaimVerdict(claim=claim, label=Label.UNSUPPORTED, confidence=1.0)

        best_ent = max(range(len(scores)), key=lambda i: scores[i].entailment)
        best_con = max(range(len(scores)), key=lambda i: scores[i].contradiction)
        e, c = scores[best_ent], scores[best_con]

        # Support wins over contradiction: one noisy "contradiction" from an unrelated window
        # must not override clear support found elsewhere.
        if e.entailment >= cfg.support_threshold:
            return ClaimVerdict(
                claim=claim,
                label=Label.SUPPORTED,
                confidence=round(e.entailment, 4),
                evidence=evidence[best_ent],
                scores=e,
            )
        if c.contradiction >= cfg.contradiction_threshold:
            return ClaimVerdict(
                claim=claim,
                label=Label.CONTRADICTED,
                confidence=round(c.contradiction, 4),
                evidence=evidence[best_con],
                scores=c,
            )
        return ClaimVerdict(
            claim=claim,
            label=Label.UNSUPPORTED,
            confidence=round(1 - e.entailment, 4),
            evidence=evidence[best_ent],
            scores=e,
        )
