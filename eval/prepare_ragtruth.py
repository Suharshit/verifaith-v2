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
from pathlib import Path


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

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            kind = halluc_type(r["hallucination_labels_processed"])
            record = {
                "id": f"ragtruth-{r['id']}",
                "answer": r["output"],
                "contexts": [r["context"]],
                "label": int(kind is None),
                "dataset": r["task_type"],
                "halluc_type": kind,
            }
            f.write(json.dumps(record) + "\n")
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
