"""Convert LLM-AggreFact (claim-level grounding benchmark) into VeriFaith's JSONL format.

    pip install datasets
    python eval/prepare_aggrefact.py --split test --out data/aggrefact.jsonl

The dataset is on the Hugging Face Hub (you may need to accept its terms and log in).
Each row is one claim + one document, label 1 = supported, so run it with --extractor sentence.
"""

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="lytang/LLM-AggreFact")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", type=Path, default=Path("data/aggrefact.jsonl"))
    args = ap.parse_args()

    from datasets import load_dataset

    ds = load_dataset(args.name, split=args.split)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as f:
        for i, row in enumerate(ds):
            f.write(
                json.dumps(
                    {
                        "id": f"{row.get('dataset', 'x')}-{i}",
                        "answer": row["claim"],
                        "contexts": [row["doc"]],
                        "label": int(row["label"]),
                        "dataset": row.get("dataset", "all"),
                    }
                )
                + "\n"
            )
    print(f"wrote {len(ds)} rows to {args.out}")


if __name__ == "__main__":
    main()
