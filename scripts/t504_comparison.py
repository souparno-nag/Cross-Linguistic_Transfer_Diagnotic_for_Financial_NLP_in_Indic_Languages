"""T-504 — the cross-encoder comparison table (CLAUDE5.md).

    python -m scripts.t504_comparison
    python -m scripts.t504_comparison --task 3 --n-boot 200      # quicker pass

Writes `reports/encoder_comparison.md` and `reports/encoder_comparison.parquet`:
in-language baselines, all nine transfer cells, the same-source MT family and
the translationese family, every encoder side by side on identical conditions.

CPU only, no model loading — it reads the prediction logs T-305 and T-302
wrote. The bootstrap is the cost: 1000 resamples per (condition, encoder,
seed), so `--n-boot` is there for a quick structural check before committing to
the full pass.

Nothing here interprets the result. Which encoder wins, and whether that is
capacity dilution or simply parameter count, is T-505.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from src.download_dataset.paths import REPO_ROOT
from src.encoder_comparison import (
    COMPARED,
    EXCLUDED,
    SEEDS,
    TASKS,
    baseline_rows,
    compare,
    condition_rows,
    render_baselines,
    render_comparison,
    render_translationese,
)
from src.env_check import require_python
from src.ids import BLOCK_NATIVE_LANG
from src.transfer import DEFAULT_N_BOOT

OUT_MD = REPO_ROOT / "reports" / "encoder_comparison.md"
OUT_PARQUET = REPO_ROOT / "reports" / "encoder_comparison.parquet"


def build(tasks, encoders, seeds, n_boot, progress=print):
    per_condition = []
    for task in tasks:
        progress(f"task {task}: summarising conditions ({n_boot} bootstrap samples)")
        per_condition.append(
            condition_rows(
                task, encoders=encoders, seeds=seeds, n_boot=n_boot, progress=progress
            )
        )
    rows = pd.concat(per_condition, ignore_index=True) if per_condition else pd.DataFrame()
    return rows, compare(rows)


def render(tasks, encoders, rows, wide) -> list[str]:
    lines = [
        "# Cross-encoder comparison — T-504",
        "",
        "Every encoder on identical conditions: the same frozen corpus, the "
        "same evaluation matrix (`configs/eval_conditions.json`), the same "
        "three seeds, the same metric definitions. Computes nothing new about "
        "any single encoder — the per-encoder numbers are T-304's bootstrap "
        "over T-306's paired join — and adds only the difference between "
        "encoders on the same items.",
        "",
        "**Confidence intervals are per seed.** T-304 bootstraps each "
        "(condition, seed) independently, so the interval shown is the mean of "
        "the three seeds' 1000-sample intervals, not a single interval over "
        "pooled seeds. The latter would be narrower and would not be earned.",
        "",
        "**Protocol.** Identical across encoders except for one measured, "
        "VRAM-forced difference: `task2_hin_mbert` uses `batch_size 8` with "
        "`grad_accum 2` where its IndicBERT sibling uses 16 with 1. The "
        "effective batch is 16 in both, and because the loss is a mean, "
        "accumulating two micro-batches of 8 produces the same gradient as one "
        "batch of 16 — the optimiser, learning rate, schedule and update count "
        "are unchanged. mBERT could not hold 16 rows of `max_len` 192 on a "
        "3.68 GiB card (T-502, reproduced in a clean process).",
        "",
        "**What is missing, and why.** "
        + " ".join(f"`{enc}` — {reason}." for enc, reason in EXCLUDED.items())
        + " Twelve of the eighteen transfer cells per task, and two of the "
        "three translationese families, are blocked for **both** encoders "
        "because Phase 2 deferred the Bengali and Telugu baselines "
        "(CLAUDE2.md); they are shown as blocked rather than dropped, so the "
        "matrix keeps its full shape.",
        "",
        "**Not interpreted here.** Which encoder wins, and whether a win "
        "reflects multilingual capacity dilution or simply parameter count, is "
        "T-505's question — mBERT has 179M parameters against IndicBERT-v2's "
        "34M, and this table cannot separate those two explanations.",
        "",
    ]
    lines += render_baselines(baseline_rows())
    for task in tasks:
        lines += render_comparison(wide, kind="transfer", task=task)
        lines += render_comparison(wide, kind="transfer_mt", task=task)
    for task in tasks:
        for lang in BLOCK_NATIVE_LANG.values():
            lines += render_translationese(task, encoders, lang)
    return lines


def main(argv: list[str] | None = None) -> int:
    require_python()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, action="append", help="repeatable; default both")
    parser.add_argument("--encoders", default=",".join(COMPARED))
    parser.add_argument("--n-boot", type=int, default=DEFAULT_N_BOOT)
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    args = parser.parse_args(argv)

    tasks = tuple(args.task) if args.task else TASKS
    encoders = tuple(e.strip() for e in args.encoders.split(",") if e.strip())
    seeds = tuple(int(s) for s in args.seeds.split(","))

    rows, wide = build(tasks, encoders, seeds, args.n_boot)
    if rows.empty:
        print("no conditions summarised", file=sys.stderr)
        return 1

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(OUT_PARQUET, index=False)
    OUT_MD.write_text("\n".join(render(tasks, encoders, rows, wide)) + "\n")
    print(f"\nwrote {OUT_MD.relative_to(REPO_ROOT)}")
    print(f"wrote {OUT_PARQUET.relative_to(REPO_ROOT)}")

    ok = rows[rows["status"] == "ok"]
    print(
        f"\n{len(ok)} of {len(rows)} (condition, encoder) cells computed; "
        f"{len(rows) - len(ok)} blocked."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
