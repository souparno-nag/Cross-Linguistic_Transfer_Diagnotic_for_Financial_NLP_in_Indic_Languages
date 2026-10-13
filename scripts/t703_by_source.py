"""T-703 — cause mix by source language (CLAUDE7.md).

    python -m scripts.t703_by_source [--n-boot N]

The question: Hindi is the worst source on both encoders (T-408). Do
Hindi-source failures have a different cause mix from Bengali- and
Telugu-source ones, or the same mix at a higher volume?

Each pair of sources is compared **only on the targets both of them reach**:
Hindi trains towards {ben, mal, tel} and Bengali towards {hin, mal, tel}, so
pooling every target would compare different target languages as much as
different sources. Native-text conditions only. Writes
`reports/phase7/by_source.{parquet,md}` and `by_source_groups.parquet`.
"""

from __future__ import annotations

import argparse
import sys

from scripts.t408_source_comparison import find_nonconverged
from src import findings as F
from src.download_dataset.paths import REPO_ROOT
from src.env_check import require_python

OUT_DIR = REPO_ROOT / "reports" / "phase7"
SOURCES = ("hin", "ben", "tel")
TARGETS = {"hin": {"ben", "mal", "tel"}, "ben": {"hin", "mal", "tel"}, "tel": {"hin", "ben", "mal"}}


def specs() -> list[dict]:
    out = []
    for task in F.TASKS:
        for i, a in enumerate(SOURCES):
            for b in SOURCES[i + 1:]:
                shared = sorted(TARGETS[a] & TARGETS[b])
                out.append({
                    "question": "T-703",
                    "name": f"Task {task}: {a} source vs {b} source, on targets {'+'.join(shared)}",
                    "a_name": f"{a}→{{{','.join(shared)}}}",
                    "b_name": f"{b}→{{{','.join(shared)}}}",
                    "fixed": {"task": [task], "kind": ["transfer"], "target": shared},
                    "a": {"source": [a]},
                    "b": {"source": [b]},
                    "replicate": "encoder",
                })
    return out


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

    keys = ["task", "encoder", "source"]
    rates = F.failure_rates(native, evaluated, keys)
    by_source = F.grouped_shares(native, keys, n_boot=args.n_boot).merge(rates, on=keys)
    details = F.run_comparisons(specs(), rows, n_boot=args.n_boot)

    intro = [
        "**Scope.** Native-text conditions only. The table below pools each source over all three of its "
        "targets, so it mixes target effects into source effects; the comparisons further down do not, "
        "because each pair of sources is compared on the targets both reach. `fail rate` is the volume "
        "half of the question: failed rows over evaluated rows.",
        "",
    ]
    body = [
        "## Cause mix by source language (all targets pooled)",
        "",
        *F.group_markdown(by_source, keys, rates),
    ]
    config_hash = F.run_config_hash(n_boot=args.n_boot)
    F.write_report("by_source", "Phase 7 — cause mix by source language (T-703)",
                   intro, body, details, by_source.assign(grouping="source"), OUT_DIR, config_hash)
    print(F.verdict_table(details).groupby(["seed_set", "verdict"]).size().to_string())
    print(f"wrote {OUT_DIR / 'by_source.md'}, config {config_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
