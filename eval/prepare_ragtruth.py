"""Convert RAGTruth (real RAG answers with human hallucination labels) to the harness format.

    pip install datasets
    python eval/prepare_ragtruth.py --per-task 200 --out data/ragtruth.jsonl

Unlike VitaminC, each answer is several sentences over real retrieved passages, so this exercises
the whole pipeline: claim extraction, the guard, sentence splitting, retrieval and scoring.
Ungated on the Hub (wandb/RAGTruth-processed).

label 1 = no hallucination span. For unfaithful answers, halluc_type records the annotated kind:
"conflict" (evident_conflict: the answer contradicts the source, which should be `contradicted`),
"baseless" (baseless_info: not in the source, which should be `unsupported`) or "both".
Sampling is balanced per task (QA, Summary, Data2txt) and per label.
"""

import argparse
import json
import random
import re
from pathlib import Path

_SENTENCE = re.compile(r"[^\n]+?(?:[.!?](?=\s|$)|\n|$)")


def sentence_spans(text: str) -> list[tuple[int, int]]:
    """(start, end) of each sentence in the ORIGINAL text, so span labels can be matched."""
    return [(m.start(), m.end()) for m in _SENTENCE.finditer(text) if m.group().strip()]


def halluc_type(labels: dict) -> str | None:
    conflict, baseless = labels.get("evident_conflict", 0), labels.get("baseless_info", 0)
    if conflict and baseless:
        return "both"
    return "conflict" if conflict else "baseless" if baseless else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="wandb/RAGTruth-processed")
    ap.add_argument("--split", default="test")
    ap.add_argument("--per-task", type=int, default=200, help="Rows per task type, half each label")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data/ragtruth.jsonl"))
    ap.add_argument(
        "--level",
        choices=["answer", "sentence"],
        default="answer",
        help="sentence: one row per answer sentence, labelled from the annotated spans. Measures "
        "claim-level accuracy directly, and runs ~8x faster than whole answers.",
    )
    ap.add_argument("--per-label", type=int, default=250, help="sentence level: rows per label")
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.name, split=args.split)
    rng = random.Random(args.seed)
    rows = []
    for task in sorted(set(ds["task_type"])):
        # Refusals and truncated outputs are not answers to verify.
        pool = [r for r in ds if r["task_type"] == task and r["quality"] == "good"]
        pos = [r for r in pool if halluc_type(r["hallucination_labels_processed"]) is None]
        neg = [r for r in pool if halluc_type(r["hallucination_labels_processed"]) is not None]
        rng.shuffle(pos)
        rng.shuffle(neg)
        half = args.per_task // 2
        rows += pos[:half] + neg[: args.per_task - half]
    rng.shuffle(rows)

    if args.level == "sentence":
        rows = sentence_rows(rows, rng, args.per_label)
    else:
        rows = [
            {
                "id": f"ragtruth-{r['id']}",
                "answer": r["output"],
                "contexts": [r["context"]],
                "label": int(halluc_type(r["hallucination_labels_processed"]) is None),
                "dataset": r["task_type"],
                "halluc_type": halluc_type(r["hallucination_labels_processed"]),
            }
            for r in rows
        ]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for record in rows:
            f.write(json.dumps(record) + "\n")
    labels = sum(r["label"] for r in rows)
    print(f"wrote {len(rows)} rows to {args.out} ({labels} faithful / {len(rows) - labels} not)")


def sentence_rows(source: list, rng: random.Random, per_label: int) -> list[dict]:
    """One row per answer sentence: label 0 if an annotated hallucination span overlaps it."""
    pos, neg = [], []
    for r in source:
        spans = [
            (s["start"], s["end"], s["label_type"]) for s in json.loads(r["hallucination_labels"])
        ]
        for i, (start, end) in enumerate(sentence_spans(r["output"])):
            hit = [t for a, b, t in spans if a < end and start < b]
            row = {
                "id": f"ragtruth-{r['id']}-s{i}",
                "answer": r["output"][start:end].strip(),
                "contexts": [r["context"]],
                "label": int(not hit),
                "dataset": r["task_type"],
                "halluc_type": (
                    "conflict" if any("Conflict" in t for t in hit) else "baseless" if hit else None
                ),
            }
            if len(row["answer"].split()) >= 4:  # fragments carry no checkable claim
                (pos if row["label"] else neg).append(row)
    rng.shuffle(pos)
    rng.shuffle(neg)
    rows = pos[:per_label] + neg[:per_label]
    rng.shuffle(rows)
    return rows


if __name__ == "__main__":
    main()
