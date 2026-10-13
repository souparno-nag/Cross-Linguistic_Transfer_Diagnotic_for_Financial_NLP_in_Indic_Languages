"""T-704 — cause mix by encoder (CLAUDE7.md).

    python -m scripts.t704_by_encoder [--n-boot N]

The question: mBERT beats IndicBERT-v2 on every measure (Phase 5). Which labels
account for the difference?

Hard rule 5's "same direction on both encoders" cannot apply to a comparison
*between* the encoders, so the replicate is the task instead: a difference is
a finding only if the CIs separate on both tasks and point the same way
(CLAUDE7.md T-704). Native-text conditions only. Writes
`reports/phase7/by_encoder.{parquet,md}` and `by_encoder_groups.parquet`.
"""

from __future__ import annotations

import argparse
import sys

from scripts.t408_source_comparison import find_nonconverged
from src import findings as F
from src.download_dataset.paths import REPO_ROOT
from src.env_check import require_python

OUT_DIR = REPO_ROOT / "reports" / "phase7"


def specs() -> list[dict]:
    return [{
        "question": "T-704",
        "name": "IndicBERT-v2 vs mBERT, all native-text conditions",
        "a_name": "indicbert-v2",
        "b_name": "mbert-base",
        "fixed": {"kind": ["transfer"]},
        "a": {"encoder": ["indicbert-v2"]},
        "b": {"encoder": ["mbert-base"]},
        "replicate": "task",
    }]


def main(argv: list[str] | None = None) -> int:
    require_python()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=F.DEFAULT_N_BOOT)
    args = parser.parse_args(argv)

    rows, evaluated, problems = F.load_all(find_nonconverged())
    if problems:
        print("\n".join(problems), file=sys.stderr)
        return 1
    native = rows[rows["kind"] == "transfer"]

    keys = ["task", "encoder"]
    rates = F.failure_rates(native, evaluated, keys)
    by_encoder = F.grouped_shares(native, keys, n_boot=args.n_boot).merge(rates, on=keys)
    details = F.run_comparisons(specs(), rows, n_boot=args.n_boot)

    intro = [
        "**Scope.** Native-text conditions only, all nine source→target pairs pooled per encoder. Both "
        "encoders are scored on the same items and conditions. mBERT fails on fewer rows, so a share here "
        "is a share of a smaller set; `fail rate` gives the volume. The replicate is the task, and task 3 "
        "is at its data ceiling (Phase 4), so a difference must survive a task with little signal.",
        "",
    ]
    body = [
        "## Cause mix by encoder",
        "",
        *F.group_markdown(by_encoder, keys, rates),
    ]
    config_hash = F.run_config_hash(n_boot=args.n_boot)
    F.write_report("by_encoder", "Phase 7 — cause mix by encoder (T-704)",
                   intro, body, details, by_encoder.assign(grouping="encoder"), OUT_DIR, config_hash)
    print(F.verdict_table(details).groupby(["seed_set", "verdict"]).size().to_string())
    print(f"wrote {OUT_DIR / 'by_encoder.md'}, config {config_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
