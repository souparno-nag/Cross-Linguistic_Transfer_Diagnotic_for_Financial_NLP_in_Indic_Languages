"""T-702 — cause mix by target language and typology (CLAUDE7.md).

    python -m scripts.t702_by_target [--n-boot N]

The question: crossing *into* Dravidian costs more than crossing back (Phase 1
T-111, Phase 4). Does the cause mix differ between Indo-Aryan→Dravidian and
Dravidian→Indo-Aryan in the same language pairs? Only Telugu is both a source
and a target, so the same-pair comparisons are hin↔tel and ben↔tel.

Native-text conditions only (`kind == "transfer"`); native vs MT text is
T-705's question. Writes `reports/phase7/by_target.{parquet,md}` and
`by_target_groups.parquet`. CPU only.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from scripts.t408_source_comparison import find_nonconverged
from src import findings as F
from src.download_dataset.paths import REPO_ROOT
from src.env_check import require_python

OUT_DIR = REPO_ROOT / "reports" / "phase7"
GAPS = REPO_ROOT / "reports" / "source_comparison.parquet"


def specs() -> list[dict]:
    out = []
    for task in F.TASKS:
        fixed = {"task": [task], "kind": ["transfer"]}
        common = {"question": "T-702", "fixed": fixed, "replicate": "encoder"}
        out += [
            {**common, "name": f"Task {task}: into Dravidian vs out of it (hin↔tel, ben↔tel)",
             "a_name": "hin→tel + ben→tel", "b_name": "tel→hin + tel→ben",
             "a": {"pair": ["hin>tel", "ben>tel"]}, "b": {"pair": ["tel>hin", "tel>ben"]}},
            {**common, "name": f"Task {task}: hin→tel vs tel→hin",
             "a_name": "hin→tel", "b_name": "tel→hin",
             "a": {"pair": ["hin>tel"]}, "b": {"pair": ["tel>hin"]}},
            {**common, "name": f"Task {task}: ben→tel vs tel→ben",
             "a_name": "ben→tel", "b_name": "tel→ben",
             "a": {"pair": ["ben>tel"]}, "b": {"pair": ["tel>ben"]}},
        ]
    return out


def gap_lines() -> list[str]:
    gaps = pd.read_parquet(GAPS)
    lines = [
        "## Transfer gap beside the cause mix",
        "",
        "Mean macro-F1 gap (source − target) per target, from `reports/source_comparison.parquet` (T-408, "
        "which leaves out seeds that never trained). Context only; not recomputed here.",
        "",
        "| task | encoder | target | mean gap | sources |",
        "|---|---|---|---|---|",
    ]
    for (task, encoder, target), g in gaps.groupby(["task", "encoder", "target"]):
        lines.append(f"| {task} | {encoder} | {target} | {g['gap_mean'].mean():.3f} | {', '.join(sorted(g['source']))} |")
    return lines + [""]


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

    target_keys, quadrant_keys = ["task", "encoder", "target"], ["task", "encoder", "quadrant"]
    target_rates = F.failure_rates(native, evaluated, target_keys)
    quadrant_rates = F.failure_rates(native, evaluated, quadrant_keys)
    by_target = F.grouped_shares(native, target_keys, n_boot=args.n_boot).merge(target_rates, on=target_keys)
    by_quadrant = F.grouped_shares(native, quadrant_keys, n_boot=args.n_boot).merge(quadrant_rates, on=quadrant_keys)
    details = F.run_comparisons(specs(), rows, n_boot=args.n_boot)

    intro = [
        "**Scope.** Native-text conditions only. Malayalam has no native split, so its conditions evaluate on "
        "MT from another block (`eval_provenance`). Hindi targets cannot receive `morphological_masking` "
        "(Hindi writes postpositions as separate words), so a 0% Hindi share is the module's blind spot, not "
        "a property of Hindi. Task 3 is at its data ceiling (Phase 4).",
        "",
    ]
    body = [
        "## Cause mix by target language",
        "",
        *F.group_markdown(by_target, target_keys, target_rates),
        "## Cause mix by typological quadrant",
        "",
        *F.group_markdown(by_quadrant, quadrant_keys, quadrant_rates),
        *gap_lines(),
    ]
    groups = pd.concat([by_target.assign(grouping="target"), by_quadrant.assign(grouping="quadrant")],
                       ignore_index=True)
    config_hash = F.run_config_hash(n_boot=args.n_boot)
    F.write_report("by_target", "Phase 7 — cause mix by target language and typology (T-702)",
                   intro, body, details, groups, OUT_DIR, config_hash)
    print(F.verdict_table(details).groupby(["seed_set", "verdict"]).size().to_string())
    print(f"wrote {OUT_DIR / 'by_target.md'}, config {config_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
