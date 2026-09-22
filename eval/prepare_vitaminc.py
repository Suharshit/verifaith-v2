"""Convert VitaminC (claim + evidence entailment) into VeriFaith's JSONL format.

    pip install datasets
    python eval/prepare_vitaminc.py --limit 1500 --out data/vitaminc.jsonl

An ungated stand-in for LLM-AggreFact, which is gated on the Hub. Each row is one claim plus
one evidence sentence, so run it with --extractor sentence.

label 1 = faithful (SUPPORTS). REFUTES and NOT ENOUGH INFO both map to 0, matching VeriFaith's
definition: a claim the context does not back is unfaithful, whether contradicted or merely absent.
"""

import argparse
import json
import random
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="tals/vitaminc")
    ap.add_argument("--split", default="test")
    ap.add_argument("--limit", type=int, default=1500, help="Balanced rows to keep in total")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("data/vitaminc.jsonl"))
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.name, split=args.split)
    pos = [r for r in ds if r["label"] == "SUPPORTS"]
    neg = [r for r in ds if r["label"] != "SUPPORTS"]

    rng = random.Random(args.seed)
    half = args.limit // 2
    rng.shuffle(pos)
    rng.shuffle(neg)
    rows = pos[:half] + neg[: args.limit - half]
    rng.shuffle(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(
                json.dumps(
                    {
                        "id": str(r["unique_id"]),
                        "answer": r["claim"],
                        "contexts": [r["evidence"]],
                        "label": int(r["label"] == "SUPPORTS"),
                        "dataset": r.get("revision_type", "all"),
                    }
                )
                + "\n"
            )
    print(f"wrote {len(rows)} rows to {args.out} ({half} supported / {len(rows) - half} not)")


if __name__ == "__main__":
    main()
