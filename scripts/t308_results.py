"""T-308 — render the results tables for the Review 3 deck.

    python -m scripts.t308_results

Reads whatever T-305/T-306 have already computed (nothing new here) and
writes `reports/transfer_results.md`: for each task, the 9-cell transfer
matrix (native family and the same-source-MT family), the four-quadrant
summary, and the translationese comparison for every language that has one.
A cell whose checkpoint doesn't exist yet is rendered as blocked, not
omitted -- the matrix's shape is always the full 9 cells.

CPU only, no model loading -- reads existing prediction logs off disk.
"""

from __future__ import annotations

from src.download_dataset.paths import REPO_ROOT
from src.ids import BLOCK_NATIVE_LANG
from src.results_tables import (
    condition_matrix,
    quadrant_summary,
    render_matrix_table,
    render_quadrant_table,
    render_translationese_table,
    translationese_comparison,
)

OUT_PATH = REPO_ROOT / "reports" / "transfer_results.md"
TASKS = (2, 3)


def render_task(task: int) -> list[str]:
    lines = [f"## Task {task}", ""]
    lines += render_matrix_table(condition_matrix(task, "transfer"), title="Transfer matrix — native/cross-block targets")
    lines += render_matrix_table(condition_matrix(task, "transfer_mt"), title="Transfer matrix — same-source MT targets")
    lines += render_quadrant_table(quadrant_summary(task, "transfer"))
    for lang in BLOCK_NATIVE_LANG.values():
        lines += render_translationese_table(translationese_comparison(task, lang), lang)
    return lines


def main() -> int:
    lines = [
        "# Transfer results (T-308)",
        "",
        "Gap = source (in-language, same items) − target, XTREME convention "
        "(§T-304). \"blocked\" means the source language's checkpoint doesn't "
        "exist yet (Bengali/Telugu baselines are deferred, CLAUDE2.md). "
        "\"within seed noise\" flags a gap smaller than its own 3-seed "
        "standard deviation (hard rule 5) -- not a finding as-is; it renders "
        "as \"—\" when only one seed is available yet, since a standard "
        "deviation over one point is undefined, not zero.",
        "",
    ]
    for task in TASKS:
        lines += render_task(task)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text("\n".join(lines) + "\n")
    print(f"wrote {OUT_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
