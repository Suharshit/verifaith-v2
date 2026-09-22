from verifaith import Label, Verifier
from verifaith.claims import SentenceExtractor
from verifaith.config import VerifierConfig
from verifaith.schemas import NLIScores
from verifaith.testing import StaticExtractor

CTX = ["The Eiffel Tower is in Paris. It was completed in 1889. It stands 330 metres tall."]


def test_pronoun_evidence_is_supported(nli):
    # v1 bug: "It stands 330 metres tall." alone could not entail a claim about the Eiffel Tower.
    v = Verifier(StaticExtractor(["The Eiffel Tower stands 330 metres tall."]), nli)
    r = v.evaluate("The Eiffel Tower stands 330 metres tall.", CTX)
    assert r.claims[0].label == Label.SUPPORTED
    assert r.verdict == "faithful"


def test_extractor_that_corrects_facts_is_caught(nli):
    # v1 bug: the LLM rewrote "1950" into "1889", so a hallucination scored as supported.
    answer = "The Eiffel Tower was completed in 1950."
    v = Verifier(StaticExtractor(["The Eiffel Tower was completed in 1889."]), nli)
    r = v.evaluate(answer, CTX)
    assert any("fell back" in w for w in r.warnings)
    assert r.claims[0].claim.text == answer and r.claims[0].claim.source == "sentence"
    assert r.claims[0].label == Label.CONTRADICTED
    assert r.verdict == "unfaithful"


def test_support_beats_noisy_contradiction():
    class Noisy:
        name = "noisy"

        def predict(self, pairs):
            # first window "contradicts" weakly-related text, second clearly supports
            return [
                NLIScores(entailment=0.0, neutral=0.1, contradiction=0.9),
                NLIScores(entailment=0.97, neutral=0.02, contradiction=0.01),
            ][: len(pairs)]

    v = Verifier(StaticExtractor(["X is Y."]), Noisy(), config=VerifierConfig(window_size=1))
    # Guard uses the same model; disable by making the answer equal to the claim and threshold 0.
    v.config.extraction_guard_threshold = 0.0
    r = v.evaluate("X is Y.", ["Unrelated stuff here. X is Y."])
    assert r.claims[0].label == Label.SUPPORTED


def test_empty_context_does_not_crash(verifier):
    r = verifier.evaluate("Python was released in 1991.", ["  "])
    assert r.claims[0].label == Label.UNSUPPORTED
    assert r.faithfulness == 0.0
    assert r.warnings


def test_no_claims(verifier):
    r = verifier.evaluate("   ", CTX)
    assert r.verdict == "no_claims" and r.counts["total"] == 0


def test_single_batched_nli_call(nli):
    v = Verifier(SentenceExtractor(), nli)
    v.evaluate("The Eiffel Tower is in Paris. It stands 330 metres tall.", CTX)
    assert nli.calls == 1  # sentence extractor skips the guard; all pairs go in one batch


def test_unsupported_claim_is_not_contradicted(verifier):
    r = verifier.evaluate("The Eiffel Tower has a restaurant.", CTX)
    assert r.claims[0].label == Label.UNSUPPORTED


def test_mixed_answer_scores_partial(verifier):
    r = verifier.evaluate("The Eiffel Tower is in Paris. The Eiffel Tower has a restaurant.", CTX)
    assert r.faithfulness == 0.5 and r.verdict == "partial"
    assert r.config["nli_model"] == "keyword-fake"
