"""T-109 — per-direction drift report.

    python -m scripts.t109_drift --task 3
    python -m scripts.t109_drift --task 3 --write-flags

Reads T-108's scores and reports the similarity distribution and below-τ count
per direction **and** per gold class. Below-τ rows are flagged
`translation_drift` and **stay in the corpus** (§4 rule 1) — a low-similarity
pair is a reported research category, not garbage.

Task 1 has no gold class, so the second breakdown is by `span_recovered`
instead: whether the annotated numeral survived is that task's equivalent of a
class label, and it is the axis T-111 cares about.

CPU only; T-108 must have run first.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from src.audit import config_hash
from src.corpus_io import is_numeral_task, read_split, write_split
from src.download_dataset.paths import REPO_ROOT
from src.ids import BLOCK_NATIVE_LANG
from src.labse_gate import VERIFICATION_ROOT, load_config

REPORT_ROOT = REPO_ROOT / "reports"
DRIFT_FLAG = "translation_drift"


def load_scores(task: int) -> pd.DataFrame:
    path = VERIFICATION_ROOT / f"task_{task}" / "labse_scores.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"no scores at {path}; run `python -m scripts.t108_labse --task {task}` first"
        )
    return pd.read_parquet(path)


def with_classes(task: int, scores: pd.DataFrame) -> pd.DataFrame:
    """Attach each pair's gold class, or task 1's span outcome."""
    parts = []
    for (block, tgt_lang), group in scores.groupby(["block_id", "tgt_lang"]):
        frame = read_split(task, block, tgt_lang)
        column = "span_recovered" if is_numeral_task(task) else "label"
        lookup = dict(zip(frame["item_id"], frame[column]))
        group = group.copy()
        group["gold_class"] = [str(lookup.get(item)) for item in group["item_id"]]
        parts.append(group)
    return pd.concat(parts, ignore_index=True)


def per_direction(scores: pd.DataFrame, tau: float) -> pd.DataFrame:
    scores = scores.assign(below=scores["labse_sim"] < tau)
    summary = (
        scores.groupby(["src_lang", "tgt_lang"])
        .agg(
            rows=("labse_sim", "size"),
            mean=("labse_sim", "mean"),
            median=("labse_sim", "median"),
            p05=("labse_sim", lambda s: s.quantile(0.05)),
            p25=("labse_sim", lambda s: s.quantile(0.25)),
            below_tau=("below", "sum"),
            below_share=("below", "mean"),
        )
        .reset_index()
        .round(4)
    )
    return summary.sort_values("below_share", ascending=False)


def per_class(scores: pd.DataFrame, tau: float) -> pd.DataFrame:
    scores = scores.assign(below=scores["labse_sim"] < tau)
    return (
        scores.groupby(["src_lang", "tgt_lang", "gold_class"])
        .agg(rows=("labse_sim", "size"), median=("labse_sim", "median"),
             below_share=("below", "mean"))
        .reset_index()
        .round(4)
    )


def apply_flags(task: int, scores: pd.DataFrame, tau: float) -> int:
    """Flag below-τ rows. Idempotent, and the rows stay (§4 rule 1)."""
    flagged = 0
    for (block, tgt_lang), group in scores.groupby(["block_id", "tgt_lang"]):
        frame = read_split(task, block, tgt_lang)
        below = set(group.loc[group["labse_sim"] < tau, "item_id"])
        frame["flags"] = [
            [f for f in flags if f != DRIFT_FLAG] + ([DRIFT_FLAG] if item in below else [])
            for item, flags in zip(frame["item_id"], frame["flags"])
        ]
        write_split(frame, task, block, tgt_lang)
        flagged += len(below)
    return flagged


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--write-flags", action="store_true")
    parser.add_argument("--tau", type=float, default=None, help="override the configured τ")
    args = parser.parse_args(argv)

    config = load_config()
    tau = args.tau if args.tau is not None else config["tau"]
    try:
        scores = with_classes(args.task, load_scores(args.task))
    except FileNotFoundError as error:
        print(error, file=sys.stderr)
        return 2

    directions = per_direction(scores, tau)
    classes = per_class(scores, tau)
    print(f"task {args.task}  τ={tau}  {len(scores)} pairs\n")
    print(directions.to_string(index=False))
    print("\nby gold class:")
    print(classes.to_string(index=False))

    report_dir = REPORT_ROOT / f"task_{args.task}"
    report_dir.mkdir(parents=True, exist_ok=True)
    directions.to_parquet(report_dir / "drift_by_direction.parquet", index=False)
    classes.to_parquet(report_dir / "drift_by_class.parquet", index=False)

    axis = "span recovered" if is_numeral_task(args.task) else "gold class"
    worst = directions.iloc[0]
    lines = [
        f"# T-109 — translation drift by direction, task {args.task}",
        "",
        f"τ = {tau}, config `{config_hash({'tau': tau, 'model': config['model']})}`, "
        f"{len(scores)} MT pairs, every pair scored (T-108).",
        "",
        "Below-τ rows carry `translation_drift` and **remain in the corpus** — §4 "
        "rule 1 makes a low-similarity pair a reported category, not garbage.",
        "",
        directions.to_markdown(index=False),
        "",
        f"Worst direction: `{worst.src_lang}→{worst.tgt_lang}` at "
        f"{worst.below_share:.1%} below τ (median {worst.median}).",
        "",
        f"## By {axis}",
        "",
        classes.to_markdown(index=False),
        "",
    ]
    if (directions["below_share"] > 0.20).any():
        over = directions[directions["below_share"] > 0.20]
        lines += [
            "## More than 20% below τ",
            "",
            "T-110 treats this as evidence that the threshold is measuring the wrong "
            "thing rather than as a quality verdict on these directions:",
            "",
            *(f"- `{r.src_lang}→{r.tgt_lang}`: {r.below_share:.1%} below τ" for r in over.itertuples()),
            "",
        ]
    (report_dir / "drift_by_direction.md").write_text("\n".join(lines))
    print(f"\nWrote {(report_dir / 'drift_by_direction.md').relative_to(REPO_ROOT)}")

    if args.write_flags:
        flagged = apply_flags(args.task, scores, tau)
        print(f"{flagged} rows flagged {DRIFT_FLAG} (kept, per rule 1)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
