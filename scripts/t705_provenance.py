"""T-705 — native vs MT target text (CLAUDE7.md).

    python -m scripts.t705_provenance [--n-boot N]

The question: how much of the transfer loss does our own translation account
for? Each target language is evaluated twice from the same source model, on
native text and on MT text (`transfer_*` vs `transfer_*_mt`), so the two
cause mixes can be compared per target.

Bengali, Hindi and Telugu only. Malayalam has no native split, so both of its
conditions evaluate on MT (from another block, and from the training block):
it is reported but is not a provenance contrast. Native Bengali also writes
Bengali digits where MT Bengali writes ASCII ones, so a Bengali difference is
provenance *and* digit system (CLAUDE4.md, Scope decisions). Writes
`reports/phase7/provenance.{parquet,md}` and `provenance_groups.parquet`.
"""

from __future__ import annotations

import argparse
import sys

from scripts.t408_source_comparison import find_nonconverged
from src import findings as F
from src.download_dataset.paths import REPO_ROOT
from src.env_check import require_python

OUT_DIR = REPO_ROOT / "reports" / "phase7"
CONTRAST_TARGETS = ("ben", "hin", "tel")
#: The two labels that speak to the translation itself.
TRANSLATION_LABELS = ("translation_drift", "provenance_specific")


def specs() -> list[dict]:
    return [
        {
            "question": "T-705",
            "name": f"Task {task}: {target} target, native vs MT text",
            "a_name": f"native {target}",
            "b_name": f"MT {target}",
            "fixed": {"task": [task], "target": [target]},
            "a": {"kind": ["transfer"]},
            "b": {"kind": ["transfer_mt"]},
            "replicate": "encoder",
        }
        for task in F.TASKS
        for target in CONTRAST_TARGETS
    ]


def translation_lines(groups) -> list[str]:
    lines = [
        "## Failures labelled as the translation's doing",
        "",
        "`translation_drift` (the MT lost the meaning) plus `provenance_specific` (wrong on this text, right on "
        "the same item in the other provenance), per condition kind. Pooled point shares; their CIs are in "
        "the tables above.",
        "",
        "| task | encoder | target | kind | translation_drift | provenance_specific | together |",
        "|---|---|---|---|---|---|---|",
    ]
    l2 = groups[groups["level"] == "L2"]
    for (task, encoder, target, kind), g in l2.groupby(["task", "encoder", "target", "kind"], sort=True):
        g = g.set_index("measure")["share"]
        parts = [g[label] for label in TRANSLATION_LABELS]
        lines.append(f"| {task} | {encoder} | {target} | {kind} | "
                     + " | ".join(F.pct(p) for p in parts) + f" | {F.pct(sum(parts))} |")
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

    keys = ["task", "encoder", "target", "kind"]
    rates = F.failure_rates(rows, evaluated, keys)
    groups = F.grouped_shares(rows, keys, n_boot=args.n_boot).merge(rates, on=keys)
    details = F.run_comparisons(specs(), rows, n_boot=args.n_boot)

    intro = [
        "**Scope.** `transfer` evaluates on native target text, `transfer_mt` on MT target text, same items "
        "and source model. Bengali native vs MT also differ in digit system. Malayalam appears in the tables "
        "but has no native text, so it is not compared.",
        "",
    ]
    body = [
        "## Cause mix by target and text provenance",
        "",
        *F.group_markdown(groups, keys, rates),
        *translation_lines(groups),
    ]
    config_hash = F.run_config_hash(n_boot=args.n_boot)
    F.write_report("provenance", "Phase 7 — native vs MT target text (T-705)",
                   intro, body, details, groups.assign(grouping="target_kind"), OUT_DIR, config_hash)
    print(F.verdict_table(details).groupby(["seed_set", "verdict"]).size().to_string())
    print(f"wrote {OUT_DIR / 'provenance.md'}, config {config_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
