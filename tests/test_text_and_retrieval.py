from verifaith import text
from verifaith.retrieval import LexicalRetriever, coverage
from verifaith.text import make_views, make_windows, split_sentences


def test_split_normalizes_whitespace_and_keeps_abbreviations():
    text = "Dr. Smith lives in the U.S. today.\n   He was born\n in 1950. It rained."
    assert split_sentences(text) == [
        "Dr. Smith lives in the U.S. today.",
        "He was born in 1950.",
        "It rained.",
    ]


def test_split_empty():
    assert split_sentences("   ") == []


def test_split_respects_markdown_lines():
    # v1 bug: newlines were collapsed first, so this whole block was ONE sentence.
    md = (
        "## Pricing\n- Basic plan: $10 per month\n* Pro plan: $25 per month\n1. Step one\n\n"
        "Refunds\nAllowed within 30 days."
    )
    assert split_sentences(md) == [
        "Pricing",
        "Basic plan: $10 per month",
        "Pro plan: $25 per month",
        "Step one",
        "Refunds",
        "Allowed within 30 days.",
    ]


def test_split_joins_hard_wrapped_lines_only():
    wrapped = "- Pro plan costs $25 per month,\n  billed annually\nThe Basic plan is free"
    assert split_sentences(wrapped) == [
        "Pro plan costs $25 per month, billed annually",
        "The Basic plan is free",
    ]


def test_markdown_table_rows_keep_their_column_names():
    table = "| Plan | Price | Seats |\n|:-----|------:|---|\n| Basic | $10 | 1 |\n| Pro | $25 | |"
    assert split_sentences(f"Our plans:\n{table}\nAsk sales.") == [
        "Our plans:",
        "Plan: Basic; Price: $10; Seats: 1.",
        "Plan: Pro; Price: $25.",
        "Ask sales.",
    ]


def test_long_text_is_chunked_so_the_model_never_truncates_it():
    sents = split_sentences(" ".join(["word"] * 300) + " the battery lasts 12 hours")
    assert all(len(s.split()) <= text.MAX_SENTENCE_WORDS for s in sents)
    assert sents[-1].endswith("the battery lasts 12 hours")


def test_windows_stay_within_the_word_budget():
    ctx = " ".join("Cats " + " ".join(["purr"] * 99) + "." for _ in range(3))
    assert [len(s.split()) for s in split_sentences(ctx)] == [100, 100, 100]
    ws = make_windows([ctx], size=3)
    assert all(len(w.text.split()) <= text.MAX_WINDOW_WORDS for w in ws)
    assert (ws[-1].start_sentence, ws[-1].end_sentence) == (1, 2)


def test_views_include_single_sentences():
    single, windows = make_views(["Cats purr. Dogs bark. Cows moo."], size=3)
    assert [w.text for w in single] == ["Cats purr.", "Dogs bark.", "Cows moo."]
    assert windows[-1].text == "Cats purr. Dogs bark. Cows moo."


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


def test_retriever_does_not_pad_with_unrelated_windows():
    # v1 bug: with more windows than k, zero-score windows filled the slots by position.
    docs = [" ".join(f"Filler sentence {i} about corporate matters." for i in range(20))]
    docs[0] += " Python was released in 1991."
    ws = make_windows(docs, size=1)
    assert [ws[i].text for i in LexicalRetriever().select("Python 1991", ws, k=8)] == [
        "Python was released in 1991."
    ]
    assert LexicalRetriever().select("The CEO resigned", ws, k=8) == []


def test_coverage_ignores_numbers_and_plurals():
    assert coverage("The tower is 500 metres tall.", "It stands 330 metre tall. The tower.") == 1.0
    assert coverage("The Louvre is in Paris.", "The Eiffel Tower is in Paris.") == 0.5
    assert coverage("1889.", "anything") == 1.0
