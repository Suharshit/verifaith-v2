"""The verification pipeline: extract -> guard -> window -> select -> NLI -> decide -> score."""

from __future__ import annotations

from verifaith.claims.base import ClaimExtractor
from verifaith.claims.sentence import SentenceExtractor
from verifaith.config import Settings, VerifierConfig
from verifaith.guard import unfaithful_extractions
from verifaith.nli.base import EntailmentModel
from verifaith.retrieval import LexicalRetriever, Retriever, coverage, select_from_views
from verifaith.schemas import (
    Claim,
    ClaimVerdict,
    EvalResult,
    Evidence,
    Label,
    NLIScores,
)
from verifaith.scoring import aggregate
from verifaith.text import make_views, split_sentences

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
        views = make_views(contexts, size=cfg.window_size)
        if not views[0]:
            warnings.append("Context contained no usable sentences; every claim is unsupported.")

        # One batched NLI call for every (evidence, claim) pair.
        plan = [
            select_from_views(self.retriever, c.text, views, cfg.max_candidates) for c in claims
        ]
        pairs = [(ev.text, c.text) for c, evs in zip(claims, plan, strict=True) for ev in evs]
        scores = self.nli.predict(pairs) if pairs else []

        verdicts, cursor = [], 0
        for claim, evs, topics in zip(claims, plan, self._topics(answer, claims), strict=True):
            chunk = scores[cursor : cursor + len(evs)]
            cursor += len(evs)
            verdicts.append(self._decide(claim, evs, chunk, topics))

        faithfulness, contradiction_rate, verdict, counts = aggregate(verdicts, cfg)
        if counts["conflicting"]:
            warnings.append(
                f"{counts['conflicting']} supported claim(s) are contradicted by other evidence: "
                "the sources disagree (see conflicting_evidence)."
            )
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
        cfg = self.config
        texts = self.extractor.extract(answer)
        source = getattr(self.extractor, "source", "llm")
        if source == "llm" and texts:
            bad = unfaithful_extractions(
                answer,
                texts,
                self.nli,
                cfg.extraction_guard_threshold,
                self.retriever,
                window_size=cfg.window_size,
                max_candidates=cfg.max_candidates,
            )
            if bad:
                # The extractor changed or invented facts. Never verify rewritten claims: fall back
                # to the answer's own sentences. All of them, not just the bad claims' sources:
                # mapping a rewritten claim back to its sentence is a guess, and a wrong guess
                # would leave the hallucinated sentence unchecked.
                warnings.append(
                    f"{len(bad)} of {len(texts)} extracted claim(s) were not faithful to the "
                    "answer; fell back to sentence-level claims."
                )
                texts, source = self._fallback.extract(answer), "sentence"
        if not texts:
            warnings.append("No factual claims found in the answer.")
        return [Claim(id=i + 1, text=t, source=source) for i, t in enumerate(texts)]

    def _topics(self, answer: str, claims: list[Claim]) -> list[list[str]]:
        """Texts that say what each claim is about, for the contradiction relevance check.

        A sentence claim like "It was completed in 1950." names its subject in an earlier answer
        sentence, so it is also judged together with the sentences before it.
        """
        sents = split_sentences(answer)
        aligned = [c.text for c in claims] == sents
        topics = []
        for i, c in enumerate(claims):
            if c.source != "sentence" or not aligned:
                topics.append([c.text])
                continue
            starts = range(max(0, i - self.config.window_size + 1), i + 1)
            topics.append([" ".join(sents[s : i + 1]) for s in starts])
        return topics

    def _decide(
        self,
        claim: Claim,
        evidence: list[Evidence],
        scores: list[NLIScores],
        topics: list[str] | None = None,
    ) -> ClaimVerdict:
        cfg = self.config
        if not scores:
            return ClaimVerdict(claim=claim, label=Label.UNSUPPORTED, confidence=1.0)

        best_ent = max(range(len(scores)), key=lambda i: scores[i].entailment)
        e = scores[best_ent]

        # Only windows about the claim may contradict it: NLI models score unrelated text as
        # contradiction ("Apples are rich in fiber" contradicts "The Eiffel Tower is 330 m tall").
        topics = topics or [claim.text]
        on_topic = [
            i
            for i, ev in enumerate(evidence)
            if max(coverage(t, ev.text) for t in topics) >= cfg.contradiction_min_overlap
        ]
        best_con = max(on_topic, key=lambda i: scores[i].contradiction, default=None)
        contradicted = (
            best_con is not None and scores[best_con].contradiction >= cfg.contradiction_threshold
        )

        # Support wins over contradiction: the claim IS grounded in a source. But when on-topic
        # evidence elsewhere contradicts it, the sources disagree, and that must be visible.
        if e.entailment >= cfg.support_threshold:
            return ClaimVerdict(
                claim=claim,
                label=Label.SUPPORTED,
                confidence=round(e.entailment, 4),
                evidence=evidence[best_ent],
                scores=e,
                conflicting_evidence=evidence[best_con]
                if contradicted and best_con != best_ent
                else None,
            )
        if contradicted:
            c = scores[best_con]
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
