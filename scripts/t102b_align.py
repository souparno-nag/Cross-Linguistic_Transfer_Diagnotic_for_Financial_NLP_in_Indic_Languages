"""T-102b — recover the item correspondence between parallel native splits.

    python -m scripts.t102b_align --task 3

Tasks 2 and 3 hold the same content in every language, but nothing in the
released files records which item is which. This rebuilds that mapping so the
splits can be joined and so overlapping items can be identified before one
language is used for training and another for evaluation (§2.2).

Task 3 aligns by exact join on its `URL` column. Task 1 is independently
sourced and is refused outright — aligning it would invent a correspondence
that does not exist.

Exits non-zero if any item fails to align across all three languages or if an
aligned item carries different labels in different languages.
"""

from __future__ import annotations

import argparse
import sys

from src.align import (
    align_by_embedding,
    align_by_key,
    alignment_config,
    summarise,
)
from src.audit import TASK_COLUMNS, config_hash
from src.download_dataset.paths import REPO_ROOT

VERIFICATION_ROOT = REPO_ROOT / "data" / "verification"
REPORT_ROOT = REPO_ROOT / "reports"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--tau", type=float, default=0.82)
    parser.add_argument("--model", default="sentence-transformers/LaBSE")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    args = parser.parse_args(argv)

    spec = TASK_COLUMNS.get(args.task)
    if spec is None:
        print(f"unknown task {args.task}", file=sys.stderr)
        return 2
    if spec.get("label") is None:
        print(
            f"Task {args.task} is independently sourced and must not be aligned. "
            "Its languages hold genuinely different content; aligning them would "
            "invent a correspondence that does not exist.",
            file=sys.stderr,
        )
        return 2

    out_dir = VERIFICATION_ROOT / f"task_{args.task}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if spec.get("join_key"):
        result = align_by_key(args.task)
    else:
        result = align_by_embedding(
            args.task,
            model_name=args.model,
            tau=args.tau,
            batch_size=args.batch_size,
            device=args.device,
            cache_dir=out_dir / "emb",
        )
    alignment_path = out_dir / "alignment.parquet"
    result.frame.to_parquet(alignment_path, index=False)

    report_dir = REPORT_ROOT / f"task_{args.task}"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "alignment.md"
    report_path.write_text(summarise(result))

    config = alignment_config(args.task, result.method)
    print(f"Wrote {alignment_path.relative_to(REPO_ROOT)}")
    print(f"Wrote {report_path.relative_to(REPO_ROOT)}")
    print(
        f"{result.aligned_items} items aligned across "
        f"{len(result.frame['lang'].unique())} languages "
        f"(config {config_hash(config)})"
    )

    failures = result.failures()
    if failures:
        print(f"\n{len(failures)} check(s) failed:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
