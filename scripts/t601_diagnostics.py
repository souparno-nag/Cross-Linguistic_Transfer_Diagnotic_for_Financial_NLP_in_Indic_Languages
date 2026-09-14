"""T-601 — run the diagnostic pipeline over every task_2 failure condition.

    python -m scripts.t601_diagnostics --task 2 --encoder indicbert-v2 [--condition NAME]

For every condition with a `data/failures/task_2/{condition}.parquet` on
disk (or just the one named by `--condition`), runs the two-stage router
(`src.diagnostics.diagnose_condition`) and writes
`data/diagnostics/task_2/{condition}.parquet` (per-instance labels) and
`data/diagnostics/audit/task_2/{condition}.parquet` (the full multi-fire
trail). Reports the `unattributed` rate per condition prominently — per
CLAUDE4.md's working agreement, it is the honest measure of this pipeline's
coverage, not something to bury in a total.

Needs a GPU (or `--device cpu`, slow) for the saliency module's model
forward passes, and a populated `configs/esg_terms.json` (see
`scripts.t603_esg_terms`) for the morphology and saliency modules to have
anything to look for.
"""

from __future__ import annotations

import argparse

import pandas as pd

from src.diagnostics import diagnose_condition, discover_conditions, write_diagnostics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--encoder", default="indicbert-v2")
    parser.add_argument("--condition", default=None, help="run just this one condition")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)

    if args.condition:
        conditions = [c for c in discover_conditions(args.task) if c["condition_id"] == args.condition]
        if not conditions:
            parser.error(f"no failures file for condition {args.condition!r} under task_{args.task}")
    else:
        conditions = discover_conditions(args.task)

    summary = []
    for entry in conditions:
        condition_id = entry["condition_id"]
        labels, audit = diagnose_condition(args.task, condition_id, args.encoder, device=args.device)
        label_path, audit_path = write_diagnostics(args.task, condition_id, labels, audit)
        n = len(labels)
        unattributed = int((labels["assigned_label"] == "unattributed").sum()) if n else 0
        print(f"{condition_id}: {n} instances -> {label_path}")
        if n:
            print(f"  {unattributed}/{n} ({unattributed / n:.1%}) unattributed")
        summary.append({
            "condition_id": condition_id, "n": n, "unattributed": unattributed,
            "unattributed_rate": unattributed / n if n else 0.0,
        })

    print()
    print(pd.DataFrame(summary).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
