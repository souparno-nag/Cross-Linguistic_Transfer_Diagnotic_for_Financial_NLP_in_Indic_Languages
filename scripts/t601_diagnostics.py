"""T-601 — run the diagnostic pipeline over every failure condition of a task.

    python -m scripts.t601_diagnostics --task {2,3} --encoder {indicbert-v2,mbert-base}
        [--condition NAME] [--device cuda|cpu] [--force] [--dry-run]

For every condition with a failures parquet on disk for the chosen encoder
(`data/failures/task_{n}/[{encoder}/]{condition}.parquet`; the default encoder
has no subdirectory), runs the two-stage router
(`src.diagnostics.diagnose_condition`) and writes per-instance labels to
`data/diagnostics/task_{n}/[{encoder}/]{condition}.parquet` and the full
multi-fire trail to `data/diagnostics/audit/task_{n}/[{encoder}/]…`.

Resumable: a condition whose label *and* audit files already exist is skipped
(`--force` redoes it), so a run interrupted by the intermittent GPU (CLAUDE.md
§3) loses at most the condition in flight. `--dry-run` lists what would run and
loads nothing. There is no silent CPU fallback: with no CUDA device the run
stops unless `--device cpu` is given, because a corpus diagnosed half on each
device is not comparable (same lesson as CLAUDE.md §8 T-106). Reports the `unattributed` rate per condition prominently — per
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

from src.data import DEFAULT_ENCODER
from src.diagnostics import (
    audit_path,
    diagnose_condition,
    diagnostics_path,
    discover_conditions,
    path_encoder,
    write_diagnostics,
)


def resolve_device(requested: str | None) -> str:
    if requested:
        return requested
    import torch  # noqa: PLC0415

    if torch.cuda.is_available():
        return "cuda"
    raise SystemExit(
        "no CUDA device and no --device given; refusing to fall back to CPU silently "
        "(pass --device cpu if that is deliberate)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--condition", default=None, help="run just this one condition")
    parser.add_argument("--device", default=None)
    parser.add_argument("--force", action="store_true", help="redo conditions that already have output")
    parser.add_argument("--dry-run", action="store_true", help="list the plan; load no model")
    args = parser.parse_args(argv)

    enc = path_encoder(args.encoder)
    conditions = discover_conditions(args.task, encoder=enc)
    if args.condition:
        conditions = [c for c in conditions if c["condition_id"] == args.condition]
        if not conditions:
            parser.error(f"no failures file for condition {args.condition!r} (task_{args.task}, {args.encoder})")
    if not conditions:
        parser.error(f"no failure sets for task_{args.task} / {args.encoder}")

    def done(cid: str) -> bool:
        return (
            diagnostics_path(args.task, cid, encoder=enc).exists()
            and audit_path(args.task, cid, encoder=enc).exists()
        )

    todo = [c for c in conditions if args.force or not done(c["condition_id"])]
    print(f"task_{args.task} / {args.encoder}: {len(conditions)} conditions, {len(todo)} to run, "
          f"{len(conditions) - len(todo)} already done")
    if args.dry_run:
        for c in todo:
            print(f"  would run {c['condition_id']}")
        return 0

    device = resolve_device(args.device)
    summary = []
    for entry in todo:
        condition_id = entry["condition_id"]
        labels, audit = diagnose_condition(args.task, condition_id, args.encoder, device=device)
        label_path, _ = write_diagnostics(args.task, condition_id, labels, audit, encoder=enc)
        n = len(labels)
        unattributed = int((labels["assigned_label"] == "unattributed").sum()) if n else 0
        not_converged = int((audit["saliency_status"] == "not_converged").sum()) if n else 0
        print(f"{condition_id}: {n} instances -> {label_path}")
        if n:
            print(f"  {unattributed}/{n} ({unattributed / n:.1%}) unattributed; "
                  f"{not_converged} saliency not_converged")
        summary.append({
            "condition_id": condition_id, "n": n, "unattributed": unattributed,
            "unattributed_rate": unattributed / n if n else 0.0, "saliency_not_converged": not_converged,
        })

    print()
    if summary:
        print(pd.DataFrame(summary).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
