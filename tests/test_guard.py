from verifaith import Label, Verifier
from verifaith.guard import number_mismatches, unfaithful_extractions
from verifaith.retrieval import LexicalRetriever
from verifaith.testing import KeywordNLI, StaticExtractor

CTX = ["The Eiffel Tower is in Paris. It was completed in 1889. It stands 330 metres tall."]


class TruncatingNLI(KeywordNLI):
    """KeywordNLI that only sees the first `limit` premise words, like a 512-token model."""

    def __init__(self, limit: int):
        super().__init__()
        self.limit = limit

    def predict(self, pairs):
        return super().predict([(" ".join(p.split()[: self.limit]), h) for p, h in pairs])


def test_equivalent_numbers_are_not_mismatches():
    answer = "Acme has three offices, 1.5 million users and a 330.0 metre tower."
    claims = ["Acme has 3 offices.", "Acme has 1,500,000 users.", "Acme's tower is 330 metres."]
    assert number_mismatches(answer, claims) == []
    assert number_mismatches("It has 1,500,000 users.", ["It has 1.5 million users."]) == []
    assert number_mismatches("It has twenty-five staff.", ["It has 25 staff."]) == []
    assert number_mismatches("It has seventeen staff.", ["It has 17 staff."]) == []


def test_changed_numbers_are_still_mismatches():
    assert number_mismatches("Completed in 1950.", ["Completed in 1889."]) == [0]
    assert number_mismatches("It has three offices.", ["It has 4 offices."]) == [0]
    assert number_mismatches("Revenue was 1.5 million.", ["Revenue was 15 million."]) == [0]


def test_claim_from_end_of_long_answer_passes_guard():
    # v1 bug: the guard used the whole answer as premise, the NLI model truncated it, and any
    # claim from the end of a long answer looked unfaithful. The limit is also shorter than a
    # 3-sentence window, so this needs the single-sentence premises too.
    filler = " ".join(f"Sentence {i} covers background." for i in range(40))
    answer = f"{filler} The museum was founded in 1901."
    claims = ["The museum was founded in 1901."]
    assert unfaithful_extractions(answer, claims, TruncatingNLI(8), 0.5, LexicalRetriever()) == []


def test_rewritten_claim_in_long_answer_is_still_caught():
    filler = " ".join(f"Sentence {i} covers background." for i in range(40))
    answer = f"{filler} The museum was founded in 1950."
    claims = ["The museum was founded in 1901."]
    assert unfaithful_extractions(answer, claims, TruncatingNLI(8), 0.5, LexicalRetriever()) == [0]


def test_one_bad_claim_falls_back_to_all_sentences():
    # The rewritten claim shares more words with sentence 1 than with its real source (sentence 2,
    # which uses a pronoun). Guessing its source would leave "1950" unchecked, so the whole answer
    # falls back to sentences.
    answer = "The Eiffel Tower is in Paris. It was completed in 1950."
    extractor = StaticExtractor(
        ["The Eiffel Tower is in Paris.", "The Eiffel Tower was completed in 1889."]
    )
    r = Verifier(extractor, KeywordNLI()).evaluate(answer, CTX)
    assert [c.claim.text for c in r.claims] == [
        "The Eiffel Tower is in Paris.",
        "It was completed in 1950.",
    ]
    assert {c.claim.source for c in r.claims} == {"sentence"}
    assert r.claims[1].label == Label.CONTRADICTED
    assert r.verdict != "faithful"
    assert any("1 of 2" in w for w in r.warnings)
