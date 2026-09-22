"""Evaluate VeriFaith on a labeled JSONL dataset with a proper dev/test split.

Each line: {"id": "...", "answer": "...", "contexts": ["..."], "label": 1}   (1 = faithful)
Optional: "dataset" (group for the breakdown) and "halluc_type" ("conflict" | "baseless" |
"both" for unfaithful rows), which measures whether contradictions come out as `contradicted`.

    python eval/prepare_ragtruth.py --per-task 200 --out data/ragtruth.jsonl
    python eval/run_eval.py data/ragtruth.jsonl --extractor sentence
    python eval/run_eval.py data/ragtruth.jsonl --extractor llm --limit 150

Reported, per split:
  default   balanced accuracy of the shipped config's verdict (faithful vs not). No tuning.
  tuned     balanced accuracy of min-claim support_score at a threshold tuned on dev ONLY.
            Tuning that threshold is tuning support_threshold for answer-level decisions.
  auroc     ranking quality of support_score (continuous) and of faithfulness (coarse).
Plus diagnostics: guard fallback rate, claims per answer, latency, and detection by type.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

from verifaith import Verifier
from verifaith.config import Settings


def split_of(item_id: str, dev_fraction: float) -> str:
    h = int(hashlib.sha256(str(item_id).encode()).hexdigest(), 16) % 10_000
    return "dev" if h < dev_fraction * 10_000 else "test"


def auroc(scores: list[float], labels: list[int]) -> float:
    pos = [s for s, y in zip(scores, labels, strict=True) if y == 1]
    neg = [s for s, y in zip(scores, labels, strict=True) if y == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def balanced_accuracy(preds: list[bool], labels: list[int]) -> float:
    tp = sum(p and y == 1 for p, y in zip(preds, labels, strict=True))
    tn = sum(not p and y == 0 for p, y in zip(preds, labels, strict=True))
    pos = sum(labels) or 1
    neg = (len(labels) - sum(labels)) or 1
    return 0.5 * (tp / pos + tn / neg)


def best_threshold(scores: list[float], labels: list[int]) -> float:
    candidates = sorted(set(scores)) + [1.01]
    return max(candidates, key=lambda t: balanced_accuracy([s >= t for s in scores], labels))


def rate(items: list[bool]) -> float:
    return round(sum(items) / len(items), 4) if items else float("nan")


def score_row(verifier: Verifier, row: dict, dev_fraction: float) -> dict:
    t0 = time.perf_counter()
    res = verifier.evaluate(row["answer"], row["contexts"])
    return {
        "id": row["id"],
        "label": int(row["label"]),
        "split": split_of(row["id"], dev_fraction),
        "group": row.get("dataset", "all"),
        "halluc_type": row.get("halluc_type"),
        # An answer is only as faithful as its weakest claim.
        "support": min((c.support_score for c in res.claims), default=1.0),
        "faithfulness": res.faithfulness,
        "verdict": res.verdict,
        "counts": res.counts,
        "fallback": any("fell back" in w for w in res.warnings),
        "seconds": round(time.perf_counter() - t0, 3),
    }


def report(subset: list[dict], thr: float, frac_thr: float) -> dict:
    y = [r["label"] for r in subset]
    return {
        "n": len(subset),
        "default_balanced_accuracy": round(
            balanced_accuracy([r["verdict"] == "faithful" for r in subset], y), 4
        ),
        "tuned_balanced_accuracy": round(
            balanced_accuracy([r["support"] >= thr for r in subset], y), 4
        ),
        # Tuning the share of claims that must be supported, instead of demanding every claim.
        "tuned_fraction_balanced_accuracy": round(
            balanced_accuracy([r["faithfulness"] >= frac_thr for r in subset], y), 4
        ),
        "auroc_support": round(auroc([r["support"] for r in subset], y), 4),
        "auroc_faithfulness": round(auroc([r["faithfulness"] for r in subset], y), 4),
    }


def by_type(subset: list[dict]) -> dict:
    """How each kind of answer is labelled: flagged at all, and flagged as contradicted.

    For faithful answers `contradicted` is the rate at which a guardrail would wrongly BLOCK.
    """
    groups: dict[str, list[dict]] = {"faithful": [r for r in subset if r["label"] == 1]}
    for t in ("conflict", "baseless", "both"):
        groups[t] = [r for r in subset if r["label"] == 0 and r["halluc_type"] == t]
    return {
        name: {
            "n": len(rows),
            "flagged": rate([r["verdict"] != "faithful" for r in rows]),
            "contradicted": rate([r["counts"]["contradicted"] > 0 for r in rows]),
        }
        for name, rows in groups.items()
        if rows
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--extractor", choices=["llm", "sentence"], default="sentence")
    ap.add_argument(
        "--nli-model",
        default=None,
        help="Override VERIFAITH_NLI_MODEL (compare NLI backends on the same data)",
    )
    ap.add_argument("--dev-fraction", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("eval/results"))
    ap.add_argument(
        "--rescore",
        type=Path,
        default=None,
        help="Recompute metrics from a records "
        "file of an earlier run, without running the models again",
    )
    args = ap.parse_args()

    lines = args.dataset.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in lines if line.strip()]
    rows = rows[: args.limit] if args.limit else rows
    config, failures = {}, 0

    if args.rescore:  # recompute metrics from a previous run's records; no model, no waiting
        records = [json.loads(line) for line in args.rescore.read_text().splitlines() if line]
        elapsed = sum(r["seconds"] for r in records)
    else:
        overrides = {"nli_model": args.nli_model} if args.nli_model else {}
        verifier = Verifier.from_settings(Settings(extractor=args.extractor, **overrides))
        config = {**verifier.config.model_dump(), "nli_model": verifier.nli.name}
        records = []
        t0 = time.perf_counter()
        for i, row in enumerate(rows, 1):
            try:
                records.append(score_row(verifier, row, args.dev_fraction))
            except Exception as e:  # count failures explicitly; never silently score them
                failures += 1
                print(f"FAILED {row.get('id')}: {type(e).__name__}")
            if i % 50 == 0:
                print(f"{i}/{len(rows)} done, {time.perf_counter() - t0:.0f}s", flush=True)
        elapsed = time.perf_counter() - t0

    dev = [r for r in records if r["split"] == "dev"]
    test = [r for r in records if r["split"] == "test"]
    dev_y = [r["label"] for r in dev]
    thr = best_threshold([r["support"] for r in dev], dev_y)
    frac_thr = best_threshold([r["faithfulness"] for r in dev], dev_y)
    seconds = sorted(r["seconds"] for r in records) or [0.0]

    summary = {
        "dataset": str(args.dataset),
        "extractor": args.extractor,
        "config": config,
        "tuned_support_threshold_from_dev": thr,
        "tuned_faithful_threshold_from_dev": frac_thr,
        "failures": failures,
        "seconds_per_item": round(elapsed / max(len(rows), 1), 3),
        "latency_p50_p95": [
            round(statistics.median(seconds), 3),
            round(seconds[int(0.95 * (len(seconds) - 1))], 3),
        ],
        "guard_fallback_rate": rate([r["fallback"] for r in records]),
        "claims_per_answer": round(
            statistics.mean(r["counts"]["total"] for r in records) if records else 0, 2
        ),
        "dev": report(dev, thr, frac_thr),
        "test": report(test, thr, frac_thr),
        "test_by_group": {
            g: report([r for r in test if r["group"] == g], thr, frac_thr)
            for g in sorted({r["group"] for r in test})
        },
        "test_by_type": by_type(test),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (args.out / f"{stamp}-summary.json").write_text(json.dumps(summary, indent=2))
    (args.out / f"{stamp}-records.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
