"""Evaluate VeriFaith on a labeled JSONL dataset with a proper dev/test split.

Each line: {"id": "...", "answer": "...", "contexts": ["..."], "label": 1}   (1 = faithful)

    python eval/run_eval.py data/aggrefact.jsonl --extractor sentence
    python eval/run_eval.py data/ragtruth.jsonl --extractor llm --limit 500

Thresholds are tuned on the dev split ONLY and reported on the untouched test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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


def balanced_accuracy(scores, labels, threshold) -> float:
    tp = sum(s >= threshold and y == 1 for s, y in zip(scores, labels, strict=True))
    tn = sum(s < threshold and y == 0 for s, y in zip(scores, labels, strict=True))
    p = sum(labels) or 1
    n = (len(labels) - sum(labels)) or 1
    return 0.5 * (tp / p + tn / n)


def best_threshold(scores, labels) -> float:
    candidates = sorted(set(scores)) + [1.01]
    return max(candidates, key=lambda t: balanced_accuracy(scores, labels, t))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", type=Path)
    ap.add_argument("--extractor", choices=["llm", "sentence"], default="sentence")
    ap.add_argument("--dev-fraction", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", type=Path, default=Path("eval/results"))
    args = ap.parse_args()

    rows = [json.loads(line) for line in args.dataset.read_text().splitlines() if line.strip()]
    rows = rows[: args.limit] if args.limit else rows
    verifier = Verifier.from_settings(Settings(extractor=args.extractor))

    records, failures = [], 0
    t0 = time.perf_counter()
    for row in rows:
        try:
            res = verifier.evaluate(row["answer"], row["contexts"])
            records.append(
                {
                    "id": row["id"],
                    "label": int(row["label"]),
                    "score": res.faithfulness,
                    "verdict": res.verdict,
                    "split": split_of(row["id"], args.dev_fraction),
                    "group": row.get("dataset", "all"),
                }
            )
        except Exception as e:  # count failures explicitly; never silently score them
            failures += 1
            print(f"FAILED {row.get('id')}: {type(e).__name__}")
    elapsed = time.perf_counter() - t0

    dev = [r for r in records if r["split"] == "dev"]
    test = [r for r in records if r["split"] == "test"]
    thr = best_threshold([r["score"] for r in dev], [r["label"] for r in dev])

    def report(subset):
        s, y = [r["score"] for r in subset], [r["label"] for r in subset]
        return {
            "n": len(subset),
            "balanced_accuracy": round(balanced_accuracy(s, y, thr), 4),
            "auroc": round(auroc(s, y), 4),
        }

    summary = {
        "dataset": str(args.dataset),
        "extractor": args.extractor,
        "threshold_from_dev": thr,
        "failures": failures,
        "seconds_per_item": round(elapsed / max(len(rows), 1), 3),
        "dev": report(dev),
        "test": report(test),
        "test_by_group": {
            g: report([r for r in test if r["group"] == g])
            for g in sorted({r["group"] for r in test})
        },
    }
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    (args.out / f"{stamp}-summary.json").write_text(json.dumps(summary, indent=2))
    (args.out / f"{stamp}-records.jsonl").write_text("\n".join(json.dumps(r) for r in records))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
