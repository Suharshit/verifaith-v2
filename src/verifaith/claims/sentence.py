from verifaith.text import split_sentences


class SentenceExtractor:
    """No-LLM fallback: every sentence of the answer is one claim. Cheap, deterministic, coarse."""

    source = "sentence"

    def extract(self, answer: str) -> list[str]:
        return split_sentences(answer)
