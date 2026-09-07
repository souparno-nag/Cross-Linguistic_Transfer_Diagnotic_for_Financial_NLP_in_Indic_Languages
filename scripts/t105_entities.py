"""T-105 — per-direction financial-entity preservation.

    python -m scripts.t105_entities --task 2                # corpus MT splits
    python -m scripts.t105_entities --task 2 --from-smoke   # T-104 sample

Reports, for every translation direction, the share of rows whose numerals,
currencies and percentages survived. Any direction below 95% is flagged.

Until T-106 writes MT into the corpus, `--from-smoke` scores the 180-row sample
T-104 produced, which is enough to validate the checker against real output.

Rows corrupted by the §3.4 placeholder or escape bugs are counted separately
and excluded from the preservation rate: their content went missing for reasons
unrelated to numeral handling, and folding them in would misattribute the cause.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from src.corpus_io import read_split
from src.download_dataset.paths import REPO_ROOT
from src.entities import compare
from src.ids import BLOCK_NATIVE_LANG, targets_for_block

REPORT_ROOT = REPO_ROOT / "reports"
THRESHOLD = 0.95


def rows_from_smoke(task: int, model: str) -> pd.DataFrame:
    path = REPORT_ROOT / f"task_{task}" / f"t104_smoke_{model}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"no smoke output at {path}; run scripts.t104_smoke --task {task} first"
        )
    frame = pd.read_parquet(path)
    return frame.rename(columns={"source": "src_text", "translation": "tgt_text"})[
        ["block", "src_lang", "tgt_lang", "src_text", "tgt_text"]
    ]


def rows_from_corpus(task: int) -> pd.DataFrame:
    """Pair each MT row with its native source on item_id."""
    records = []
    for block, native in BLOCK_NATIVE_LANG.items():
        source = read_split(task, block, native).set_index("item_id")
        for target in targets_for_block(block):
            try:
                mt = read_split(task, block, target)
            except FileNotFoundError:
                continue
            for _, row in mt.iterrows():
                if row["item_id"] not in source.index:
                    continue
                records.append(
                    {
                        "block": block,
                        "src_lang": native,
                        "tgt_lang": target,
                        "src_text": source.loc[row["item_id"], "text"],
                        "tgt_text": row["text"],
                        "item_id": row["item_id"],
                    }
                )
    return pd.DataFrame(records)


def score(frame: pd.DataFrame) -> pd.DataFrame:
    results = []
    for _, row in frame.iterrows():
        outcome = compare(str(row["src_text"]), str(row["tgt_text"]))
        results.append(
            {
                **{k: row[k] for k in row.index},
                "preserved": outcome.preserved,
                "dropped": len(outcome.dropped),
                "altered": len(outcome.added),
                "currency_lost": len(outcome.currency_lost),
                "percent_lost": outcome.percent_lost,
                "corrupted": bool(outcome.corrupted),
                "flags": outcome.flags(),
            }
        )
    return pd.DataFrame(results)


def per_direction(scored: pd.DataFrame) -> pd.DataFrame:
    clean = scored[~scored["corrupted"]]
    grouped = (
        clean.groupby(["src_lang", "tgt_lang"])
        .agg(
            rows=("preserved", "size"),
            preserved=("preserved", "mean"),
            dropped=("dropped", "sum"),
            altered=("altered", "sum"),
            currency_lost=("currency_lost", "sum"),
            percent_lost=("percent_lost", "sum"),
        )
        .reset_index()
    )
    corrupt = (
        scored.groupby(["src_lang", "tgt_lang"])["corrupted"].sum().reset_index()
    )
    grouped = grouped.merge(corrupt, on=["src_lang", "tgt_lang"], how="left")
    grouped["preserved"] = grouped["preserved"].round(4)
    grouped["below_threshold"] = grouped["preserved"] < THRESHOLD
    return grouped.sort_values("preserved")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--from-smoke", action="store_true")
    parser.add_argument("--model", default="320M", help="smoke model label")
    args = parser.parse_args(argv)

    frame = (
        rows_from_smoke(args.task, args.model)
        if args.from_smoke
        else rows_from_corpus(args.task)
    )
    if frame.empty:
        print(
            f"No translated rows for task {args.task}. T-106 has not run; use "
            "--from-smoke to score the T-104 sample instead.",
            file=sys.stderr,
        )
        return 2

    scored = score(frame)
    summary = per_direction(scored)

    report_dir = REPORT_ROOT / f"task_{args.task}"
    report_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_smoke" if args.from_smoke else ""
    scored.to_parquet(report_dir / f"entity_preservation{suffix}.parquet", index=False)

    print(summary.to_string(index=False))
    failing = summary[summary["below_threshold"]]
    corrupted = int(scored["corrupted"].sum())
    print(
        f"\n{len(scored)} rows scored; {corrupted} excluded as corrupted "
        f"({corrupted / len(scored):.1%})"
    )

    lines = [
        f"# T-105 — entity preservation, task {args.task}",
        "",
        f"Source: {'T-104 smoke sample' if args.from_smoke else 'corpus MT splits'}"
        f" · {len(scored)} rows · threshold {THRESHOLD:.0%}",
        "",
        summary.to_markdown(index=False),
        "",
        f"{corrupted} rows ({corrupted / len(scored):.1%}) carry the §3.4 placeholder "
        "or escape corruption and are excluded from the rate above — their content was "
        "lost for reasons unrelated to numeral handling.",
        "",
    ]
    if len(failing):
        lines += [
            "## Directions below threshold",
            "",
            *(
                f"- `{r.src_lang}→{r.tgt_lang}`: {r.preserved:.1%} preserved, "
                f"{int(r.dropped)} dropped, {int(r.altered)} altered"
                for r in failing.itertuples()
            ),
            "",
        ]
    (report_dir / f"entity_preservation{suffix}.md").write_text("\n".join(lines))
    print(f"Wrote {(report_dir / f'entity_preservation{suffix}.md').relative_to(REPO_ROOT)}")

    if len(failing):
        print(f"\n{len(failing)} direction(s) below {THRESHOLD:.0%}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
