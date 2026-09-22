import importlib.util
from pathlib import Path

from verifaith import Label, Verifier
from verifaith.claims import SentenceExtractor
from verifaith.testing import KeywordNLI

_spec = importlib.util.spec_from_file_location(
    "run_eval", Path(__file__).parents[1] / "eval" / "run_eval.py"
)
run_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_eval)

CTX = ["The Eiffel Tower is in Paris. It was completed in 1889. It stands 330 metres tall."]


def test_every_verdict_carries_its_support_score():
    v = Verifier(SentenceExtractor(), KeywordNLI())
    r = v.evaluate("The Eiffel Tower is in Paris. The Eiffel Tower has a restaurant.", CTX)
    assert [c.label for c in r.claims] == [Label.SUPPORTED, Label.UNSUPPORTED]
    assert r.claims[0].support_score == 0.95 and r.claims[1].support_score == 0.03


def test_answer_score_is_its_weakest_claim():
    v = Verifier(SentenceExtractor(), KeywordNLI())
    row = {"id": "x", "label": 0, "answer": "The Eiffel Tower is in Paris. It has a restaurant."}
    rec = run_eval.score_row(v, {**row, "contexts": CTX}, dev_fraction=0.3)
    assert rec["support"] == 0.03 and rec["verdict"] != "faithful" and not rec["fallback"]


def test_metrics():
    assert run_eval.auroc([0.9, 0.8, 0.1], [1, 1, 0]) == 1.0
    assert run_eval.auroc([0.5, 0.5], [1, 0]) == 0.5
    assert run_eval.balanced_accuracy([True, False, True, True], [1, 0, 0, 0]) == 0.5 * (1 + 1 / 3)
    assert run_eval.best_threshold([0.9, 0.8, 0.2, 0.1], [1, 1, 0, 0]) == 0.8


def test_report_includes_tuned_fraction():
    rows = [
        {"label": 1, "verdict": "partial", "support": 0.9, "faithfulness": 0.8},
        {"label": 0, "verdict": "partial", "support": 0.1, "faithfulness": 0.2},
    ]
    out = run_eval.report(rows, thr=0.5, frac_thr=0.5)
    assert out["default_balanced_accuracy"] == 0.5
    assert out["tuned_balanced_accuracy"] == 1.0 and out["tuned_fraction_balanced_accuracy"] == 1.0


def test_by_type_reports_wrongful_blocks_on_faithful_answers():
    rec = {"verdict": "faithful", "counts": {"contradicted": 0}, "halluc_type": None}
    rows = [
        {**rec, "label": 1},
        {**rec, "label": 1, "verdict": "partial", "counts": {"contradicted": 1}},
        {
            **rec,
            "label": 0,
            "halluc_type": "conflict",
            "verdict": "unfaithful",
            "counts": {"contradicted": 1},
        },
    ]
    out = run_eval.by_type(rows)
    assert out["faithful"] == {"n": 2, "flagged": 0.5, "contradicted": 0.5}
    assert out["conflict"] == {"n": 1, "flagged": 1.0, "contradicted": 1.0}
