"""T-305 — run every runnable evaluation condition, 3 seeds each.

    python -m scripts.t305_run_conditions --task 2
    python -m scripts.t305_run_conditions --task 2 --plan-only

Reads `configs/eval_conditions.json` and, for every condition whose model
checkpoint (`task{N}_{lang}_indicbert_seed{S}`) exists on disk, predicts every
arm, writes its prediction log (T-302) and logs a bootstrapped macro-F1 row to
`experiments.csv` (T-304). A condition whose checkpoint does not exist yet
(Bengali/Telugu baselines are still deferred per CLAUDE2.md) is reported as
blocked rather than silently skipped.

Done, per T-305: every *runnable* condition has a result row in
experiments.csv; `--plan-only` prints what is runnable today without loading
any model.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from src.evaluate import plan, run_all


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)

    entries = plan(args.task)
    print(pd.DataFrame(entries).to_string(index=False))
    n_runnable = sum(e["runnable"] for e in entries)
    print(f"\n{n_runnable}/{len(entries)} (condition, seed) pairs are runnable today.")

    if args.plan_only:
        return 0

    result = run_all(
        args.task, device=args.device, batch_size=args.batch_size, n_boot=args.n_boot,
        progress=print,
    )
    print(f"\nran {len(result['ran'])} rows, blocked {len(result['blocked'])} (condition, seed) pairs.")
    if result["blocked"]:
        blocked_conditions = sorted({b["condition"] for b in result["blocked"]})
        print(
            f"blocked conditions (missing checkpoint): {blocked_conditions}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
