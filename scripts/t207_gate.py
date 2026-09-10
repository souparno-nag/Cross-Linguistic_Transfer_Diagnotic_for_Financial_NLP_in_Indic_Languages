"""T-207 — baseline validation gate.

    python -m scripts.t207_gate            # verdict + reports/baseline_gate.md
    python -m scripts.t207_gate --check    # exit non-zero if the gate fails, print only

Compares T-206's in-language baselines against IndicFinNLP's published IndicBERT
numbers. A config passes if our mean macro-F1 is within tolerance of the
published value or above it; a config below tolerance passes only with a written
`diagnosis` in `configs/published_baselines.json`.

**Nothing in Phase 3 starts until this exits 0.**
"""

from __future__ import annotations

import argparse
import sys

from src.baseline_gate import evaluate, gate_passes, load_published, our_results
from src.download_dataset.paths import REPO_ROOT

REPORT = REPO_ROOT / "reports" / "baseline_gate.md"


def _render(published: dict, rows: list[dict]) -> str:
    tol = published.get("tolerance_f1", 0.02)
    lines = [
        "# Baseline validation gate — T-207",
        "",
        f"Our macro-F1 (T-206, mean over seeds) vs {published['source']}",
        "",
        f"Tolerance: ±{tol * 100:.0f} F1 points. A config below that passes only with a "
        "written diagnosis.",
        "",
        "| task | config | published | ours (mean ± std) | gap | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        ours = (
            "—"
            if r["our_macro_f1_mean"] is None
            else f"{r['our_macro_f1_mean']:.4f} ± {r['our_macro_f1_std']:.4f}"
        )
        gap = "—" if r["gap"] is None else f"{r['gap']:+.4f}"
        mark = "PASS" if r["passed"] else "FAIL"
        lines.append(
            f"| {r['task']} | {r['config']} | {r['published_macro_f1']:.2f} | "
            f"{ours} | {gap} | {mark} — {r['status']} |"
        )
    lines.append("")
    for r in rows:
        if r["note"]:
            lines += [f"**{r['config']} — note.** {r['note']}", ""]
        if r["diagnosis"]:
            lines += [f"**{r['config']} — diagnosis of the gap.** {r['diagnosis']}", ""]

    lines += [
        "## Verdict",
        "",
        ("**GATE PASSES.** Phase 3 may proceed." if gate_passes(rows)
         else "**GATE FAILS.** Resolve the FAIL rows — improve the run or write a "
              "diagnosis — before any Phase 3 work."),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="verdict only, no file write")
    args = parser.parse_args()

    published = load_published()
    rows = evaluate(published, our_results())

    for r in rows:
        ours = "—" if r["our_macro_f1_mean"] is None else f"{r['our_macro_f1_mean']:.4f}"
        mark = "PASS" if r["passed"] else "FAIL"
        print(
            f"  task {r['task']:<2} {r['config']:<28} "
            f"ours {ours}  pub {r['published_macro_f1']:.2f}  → {mark} ({r['status']})"
        )

    passed = gate_passes(rows)
    if not args.check:
        REPORT.parent.mkdir(parents=True, exist_ok=True)
        REPORT.write_text(_render(published, rows))
        print(f"\nwrote {REPORT}")

    print("\nGATE PASSES" if passed else "\nGATE FAILS")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
