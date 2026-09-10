"""T-113 — build and validate the evaluation-condition matrix.

    python -m scripts.t113_conditions --task 2
    python -m scripts.t113_conditions --task 2 --validate-only

Writes `configs/eval_conditions.json` (one entry per task) and checks that no
split is both training source and evaluation target within a condition, and —
the check that actually matters for tasks 2 and 3 — that no *item* is.

CPU only; needs the corpus splits to exist so item sets can be read.
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from src.conditions import DEFAULT_SEED, build, partition_of, split_name, validate
from src.corpus_io import read_split
from src.download_dataset.paths import REPO_ROOT
from src.ids import BLOCK_NATIVE_LANG, targets_for_block

CONFIG_PATH = REPO_ROOT / "configs" / "eval_conditions.json"


def item_sets(task: int) -> dict[str, set]:
    out = {}
    for block, native in BLOCK_NATIVE_LANG.items():
        for lang in (native, *targets_for_block(block)):
            try:
                frame = read_split(task, block, lang)
            except FileNotFoundError:
                continue
            out[split_name(task, block, lang)] = set(frame["item_id"])
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)

    matrix = build(args.task, seed=args.seed)
    sets = item_sets(args.task)
    failures = validate(matrix, sets)

    rows = [
        {
            "condition": c["name"],
            "kind": c["kind"],
            "quadrant": c["quadrant"] or "",
            "train": c["train"] or "",
            "eval": c["eval"] if isinstance(c["eval"], str) else f"{len(c['eval'])} arms",
            "items": f"{c['train_items'] or '-'} / {c['eval_items']}",
        }
        for c in matrix["conditions"]
    ]
    print(f"task {args.task}  seed {args.seed}  partitioned {matrix['partitioned']}\n")
    print(pd.DataFrame(rows).to_string(index=False))

    if matrix["partitioned"] and sets:
        sample = next(iter(sets.values()))
        train_half = sum(1 for i in sample if partition_of(i, args.seed) == "train")
        print(
            f"\nitem partition: {train_half}/{len(sample)} items in the training half "
            f"({train_half / len(sample):.1%}), identical across every language "
            "because it is hashed from item_id"
        )

    if failures:
        print("\nVALIDATION FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1
    print("\nvalidator: no split and no item is both trained on and evaluated on.")

    if args.validate_only:
        return 0

    existing = json.loads(CONFIG_PATH.read_text()) if CONFIG_PATH.exists() else {}
    existing[f"task_{args.task}"] = matrix
    CONFIG_PATH.write_text(json.dumps(existing, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {CONFIG_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
