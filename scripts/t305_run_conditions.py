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

from src.data import DEFAULT_ENCODER
from src.evaluate import plan, run_all


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--encoder",
        default=DEFAULT_ENCODER,
        help="which encoder's checkpoints to evaluate (CLAUDE5.md). Non-default "
        "encoders write their prediction logs under their own path level.",
    )
    args = parser.parse_args(argv)

    entries = plan(args.task, encoder=args.encoder)
    print(pd.DataFrame(entries).to_string(index=False))
    n_runnable = sum(e["runnable"] for e in entries)
    print(f"\n{n_runnable}/{len(entries)} (condition, seed) pairs are runnable today.")

    if args.plan_only:
        return 0

    result = run_all(
        args.task, device=args.device, batch_size=args.batch_size, n_boot=args.n_boot,
        progress=print, encoder=args.encoder,
    )
    print(
        f"\nran {len(result['ran'])} rows, blocked {len(result['blocked'])} "
        f"(condition, seed) pairs, failed {len(result['failed'])}."
    )
    if result["blocked"]:
        blocked_conditions = sorted({b["condition"] for b in result["blocked"]})
        print(
            f"blocked conditions (missing checkpoint): {blocked_conditions}",
            file=sys.stderr,
        )
    if result["failed"]:
        # Distinct from blocked: blocked means the checkpoint does not exist
        # yet and the cell is simply not run. Failed means it exists and the
        # pair broke, which needs looking at rather than waiting for.
        print(f"\n{len(result['failed'])} (condition, seed) pair(s) FAILED:", file=sys.stderr)
        for item in result["failed"]:
            print(
                f"  {item['condition']} seed{item['seed']} ({item['run_id']}): "
                f"{item['error']}",
                file=sys.stderr,
            )
        print(
            "\nRe-running is safe and idempotent: prediction logs replace rows "
            "keyed on (item_id, seed, run_id, block_id, lang, origin), so the "
            "pairs that succeeded are not redone incorrectly. If the errors are "
            "network ones, set HF_HUB_OFFLINE=1 first — everything is cached "
            "once a baseline has trained.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
