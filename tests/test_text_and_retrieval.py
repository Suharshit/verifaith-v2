from verifaith.retrieval import LexicalRetriever
from verifaith.text import make_windows, split_sentences


def test_split_normalizes_whitespace_and_keeps_abbreviations():
    text = "Dr. Smith lives in the U.S. today.\n   He was born\n in 1950. It rained."
    assert split_sentences(text) == [
        "Dr. Smith lives in the U.S. today.",
        "He was born in 1950.",
        "It rained.",
    ]


def test_split_empty():
    assert split_sentences("   ") == []


def test_windows_carry_the_antecedent():
    ws = make_windows(["The Eiffel Tower is in Paris. It stands 330 metres tall."], size=3)
    assert ws[-1].text == "The Eiffel Tower is in Paris. It stands 330 metres tall."
    assert (ws[-1].start_sentence, ws[-1].end_sentence) == (0, 1)


def test_retriever_never_pads_when_fewer_windows_than_k():
    ws = make_windows(["One sentence here. Another one there."])
    assert LexicalRetriever().select("claim", ws, k=8) == [0, 1]


def test_retriever_ranks_relevant_window_first():
    docs = ["Cats sleep a lot. Dogs bark loudly. Python was released in 1991. Rain is wet."]
    ws = make_windows(docs, size=1)
    top = LexicalRetriever().select("Python release year 1991", ws, k=1)
    assert ws[top[0]].text == "Python was released in 1991."
