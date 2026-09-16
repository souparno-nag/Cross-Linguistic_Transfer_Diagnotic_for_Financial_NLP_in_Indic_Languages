"""T-408 — source-selection comparison table.

    python -m scripts.t408_source_comparison
    python -m scripts.t408_source_comparison --n-boot 200      # faster draft

Writes `reports/source_comparison.{md,parquet}`: all source blocks × all
encoders × all targets, with gaps, seed spreads and the two questions
CLAUDE6.md asks of this task — which source transfers best into each target,
and whether typological proximity predicts it.

CPU only; reads T-305's prediction logs off disk and loads no model. The
bootstrap makes it slow rather than heavy — allow a few minutes.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from src import source_comparison as SC
from src.convergence import converged, history_for
from src.baseline_gate import BASELINES_PARQUET
from src.download_dataset.paths import REPO_ROOT
from src.env_check import require_python

REPORT_DIR = REPO_ROOT / "reports"
TASKS = (2, 3)


def find_nonconverged(baselines_path=BASELINES_PARQUET) -> dict:
    """Runs that never fit their training data, by run-id prefix.

    **Read back out of T-206's report, not re-derived from the checkpoints.**
    Two reasons, and the second is why this function was rewritten:

    * It is the same artefact the baseline table already reports convergence
      from, so the two documents cannot disagree about which seeds trained.
    * Deriving it here meant `torch.load`ing every checkpoint — model weights
      and optimiser state and all — purely to read an epoch history, 36 times.
      That took longer than every bootstrap in this report combined. Reading
      the parquet takes about two milliseconds.

    Falls back to the checkpoints when the column is absent, so a report
    written before T-402 added it still works.
    """
    import pandas as pd

    frame = pd.read_parquet(baselines_path)
    out: dict[str, set[int]] = {}
    if "converged" in frame.columns:
        for _, row in frame.iterrows():
            if row["converged"] is False or row["converged"] == 0:
                prefix = str(row["run_id"]).rsplit("_seed", 1)[0]
                out.setdefault(prefix, set()).add(int(row["seed"]))
        return out

    for _, row in frame.iterrows():  # legacy report: fall back to checkpoints
        run_id = str(row["run_id"])
        if converged(history_for(run_id)) is False:
            prefix, seed = run_id.rsplit("_seed", 1)
            out.setdefault(prefix, set()).add(int(seed))
    return out


def _fmt(value, spec=".4f", dash="—"):
    return dash if value is None else format(value, spec)


def main(argv: list[str] | None = None) -> int:
    require_python()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--out", default=str(REPORT_DIR))
    args = parser.parse_args(argv)

    nonconverged = find_nonconverged()
    if nonconverged:
        print("excluding seeds that never fit their training data:")
        for prefix, seeds in sorted(nonconverged.items()):
            print(f"  {prefix}: seed(s) {sorted(seeds)}")
        print()

    all_rows = []
    for task in TASKS:
        print(f"task {task}: computing cells…")
        all_rows += SC.cells(task, nonconverged=nonconverged, n_boot=args.n_boot)
    frame = pd.DataFrame(all_rows)

    lines = [
        "# Source-selection comparison — T-408",
        "",
        "All source blocks × all encoders × all targets. The gap is "
        "`source − target` macro-F1 on the **same items** (T-304's paired "
        "bootstrap), so a lower gap is a better transfer source and a source "
        "that simply starts higher is not flattered by where it started.",
        "",
        "Every number is read back out of T-305's prediction logs through "
        "T-308's own summariser. Nothing here is recomputed by a second route, "
        "because a summary that derives its own numbers can disagree with the "
        "report it summarises, silently.",
        "",
    ]

    if nonconverged:
        lines += [
            "## Seeds excluded, and why this is the one place that is allowed",
            "",
            "A model that never left its initialisation scores equally badly in "
            "its own language and in every other, so its transfer gap is near "
            "**zero** and it ranks as the *best* source in the table. That is "
            "not a subtle distortion: it inverted task 3's Bengali comparison "
            "at T-404. These seeds are therefore left out of the means below "
            "and named here rather than hidden (`src/convergence.py`).",
            "",
            "| run | excluded seeds |",
            "|---|---|",
        ]
        for prefix, seeds in sorted(nonconverged.items()):
            lines.append(f"| `{prefix}` | {sorted(seeds)} |")
        lines.append("")

    for encoder, reason in sorted(SC.EXCLUDED_ENCODERS.items()):
        lines += [f"**`{encoder}` is absent from every table below.** {reason}", ""]

    for task in TASKS:
        rows = [r for r in all_rows if r["task"] == task]
        lines += [f"## Task {task}", "", "### Transfer gap by source and target", "",
                  "| encoder | source | target | same family | n seeds | gap (mean±std) "
                  "| source | target score |", "|---|---|---|---|---|---|---|---|"]
        for r in sorted(rows, key=lambda r: (r["encoder"], r["source"], r["target"])):
            if r["status"] != "ok":
                lines.append(
                    f"| {r['encoder']} | {r['source']} | {r['target']} | "
                    f"{'yes' if r['same_family'] else 'no'} | 0 | blocked | — | — |"
                )
                continue
            lines.append(
                f"| {r['encoder']} | {r['source']} | {r['target']} | "
                f"{'yes' if r['same_family'] else 'no'} | {r['n_seeds']} | "
                f"{_fmt(r['gap_mean'])} ± {_fmt(r['gap_std'])} | "
                f"{_fmt(r['source_mean'])} | {_fmt(r['target_mean'])} |"
            )

        lines += ["", "### Which source is the best overall?", "",
                  "| encoder | source | n targets | mean gap | best into | worst into |",
                  "|---|---|---|---|---|---|"]
        for r in SC.rank_sources(rows):
            lines.append(
                f"| {r['encoder']} | **{r['source']}** | {r['n_targets']} | "
                f"{_fmt(r['mean_gap'])} | {r['best_target']} | {r['worst_target']} |"
            )

        lines += ["", "### Best source per target — does typological proximity predict it?", "",
                  "| encoder | target | best source | ranking (source: gap) | "
                  "same-family source available | proximity predicts? |",
                  "|---|---|---|---|---|---|"]
        for r in SC.best_source_per_target(rows):
            ranking = ", ".join(f"{s}: {g:.4f}" for s, g in r["ranking"])
            predicts = {True: "**yes**", False: "**no**", None: "n/a"}[r["proximity_predicts"]]
            lines.append(
                f"| {r['encoder']} | {r['target']} | **{r['best_source']}** | {ranking} "
                f"| {'yes' if r['same_family_available'] else 'no'} | {predicts} |"
            )

        lines += ["", "### Is transfer symmetric?", "",
                  "A difference smaller than the two cells' combined seed spread is not "
                  "a finding (`hard rule 5`, applied to the comparison rather than to "
                  "either cell).", "",
                  "| encoder | pair | forward | reverse | difference | harder direction "
                  "| beats noise? |", "|---|---|---|---|---|---|---|"]
        for r in SC.asymmetries(rows):
            lines.append(
                f"| {r['encoder']} | {r['pair']} | {r['forward']} "
                f"{_fmt(r['forward_gap'])} | {r['reverse']} {_fmt(r['reverse_gap'])} | "
                f"{_fmt(r['difference'])} | **{r['harder_direction']}** | "
                f"{'yes' if r['beats_noise'] else 'no'} |"
            )

        lines += ["", "### Does the cost sit in the target language?", "",
                  "Pooled over source family, which is the comparison that separates "
                  "\"this target is hard\" from \"this pairing is distant\". Phase 1's "
                  "T-111 found the same shape in the machine translation itself.", "",
                  "| encoder | target family | n cells | mean gap |", "|---|---|---|---|"]
        for r in SC.target_family_effect(rows):
            lines.append(
                f"| {r['encoder']} | {r['target_family']} | {r['n_cells']} | "
                f"{_fmt(r['mean_gap'])} |"
            )
        lines.append("")

    out_dir = REPO_ROOT / args.out if not str(args.out).startswith("/") else None
    out_dir = out_dir or __import__("pathlib").Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_dir / "source_comparison.parquet", index=False)
    (out_dir / "source_comparison.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out_dir / 'source_comparison.md'}")
    print(f"wrote {out_dir / 'source_comparison.parquet'} ({len(frame)} cells)")

    ok = int((frame["status"] == "ok").sum())
    if ok == 0:
        print("no computable cells", file=sys.stderr)
        return 1
    print(f"{ok}/{len(frame)} cells computed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
