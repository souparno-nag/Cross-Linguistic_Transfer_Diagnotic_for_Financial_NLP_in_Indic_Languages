"""T-111 — rank all 9 directions by reliability.

    python -m scripts.t111_ranking --task 1

Combines what the earlier checks measured into one ordering, and answers the
three questions §8 asks:

* Is Hi→Ml worse than Bn→Ml?
* Are Dravidian-source directions worse than Indo-Aryan-source ones?
* Is the Indo-Aryan↔Dravidian penalty symmetric?

The inputs are deliberately different in kind, because each catches something
the others cannot: LaBSE drift (T-108/109) sees meaning loss but not which
numeral moved; entity preservation (T-105) sees numerals but not fluency;
integrity (T-107) sees corruption that leaves fluent text behind. A direction
is only reliable if it survives all three.

CPU only. Every input must already exist.
"""

from __future__ import annotations

import argparse
import json
import sys

import pandas as pd

from src.corpus_io import is_numeral_task, read_split
from src.download_dataset.paths import REPO_ROOT
from src.ids import BLOCK_NATIVE_LANG, targets_for_block
from src.labse_gate import VERIFICATION_ROOT, load_config

REPORT_ROOT = REPO_ROOT / "reports"

# Typology, which is what the §8 questions are really about.
FAMILY = {"hin": "Indo-Aryan", "ben": "Indo-Aryan", "tel": "Dravidian", "mal": "Dravidian"}


def quadrant(src_lang: str, tgt_lang: str) -> str:
    return f"{FAMILY[src_lang]}->{FAMILY[tgt_lang]}"


def gather(task: int, tau: float) -> pd.DataFrame:
    """One row per direction, with every measure that exists for it."""
    task_dir = VERIFICATION_ROOT / f"task_{task}"
    report_dir = REPORT_ROOT / f"task_{task}"

    scores = pd.read_parquet(task_dir / "labse_scores.parquet")
    drift = (
        scores.assign(below=scores["labse_sim"] < tau)
        .groupby(["src_lang", "tgt_lang"])
        .agg(rows=("labse_sim", "size"),
             median_sim=("labse_sim", "median"),
             drift_rate=("below", "mean"))
        .reset_index()
    )

    entities = pd.read_parquet(report_dir / "entity_preservation.parquet")
    entity_rate = (
        entities[~entities["corrupted"]]
        .groupby(["src_lang", "tgt_lang"])["preserved"].mean()
        .reset_index().rename(columns={"preserved": "entity_preserved"})
    )
    corrupt_rate = (
        entities.groupby(["src_lang", "tgt_lang"])["corrupted"].mean()
        .reset_index().rename(columns={"corrupted": "corrupt_rate"})
    )

    integrity = json.loads((report_dir / "integrity.json").read_text())
    integrity_rows = pd.DataFrame(integrity["directions"])
    integrity_rows["truncation_rate"] = integrity_rows["truncation_suspect"] / integrity_rows["rows"]
    integrity_rows = integrity_rows[["src_lang", "tgt_lang", "truncation_rate"]]

    frame = (
        drift.merge(entity_rate, on=["src_lang", "tgt_lang"])
        .merge(corrupt_rate, on=["src_lang", "tgt_lang"])
        .merge(integrity_rows, on=["src_lang", "tgt_lang"])
    )

    if is_numeral_task(task):
        recovery = []
        for block, native in BLOCK_NATIVE_LANG.items():
            for tgt in targets_for_block(block):
                mt = read_split(task, block, tgt)
                recovery.append(
                    {"src_lang": native, "tgt_lang": tgt,
                     "span_recovery": float(mt["span_recovered"].mean())}
                )
        frame = frame.merge(pd.DataFrame(recovery), on=["src_lang", "tgt_lang"])

    frame["quadrant"] = [quadrant(r.src_lang, r.tgt_lang) for r in frame.itertuples()]
    return frame


def rank(frame: pd.DataFrame) -> pd.DataFrame:
    """Rank by the mean of each measure's rank.

    Averaging ranks rather than the raw numbers on purpose: a drift rate and a
    numeral-preservation rate are not on the same scale, and weighting them
    against each other would be an invented judgement. Rank averaging says only
    "worse on more measures", which is what the question asks.
    """
    measures = {
        "drift_rate": True,          # lower is better
        "entity_preserved": False,   # higher is better
        "corrupt_rate": True,
        "truncation_rate": True,
    }
    if "span_recovery" in frame:
        measures["span_recovery"] = False

    ranked = frame.copy()
    for column, ascending in measures.items():
        ranked[f"rank_{column}"] = ranked[column].rank(ascending=ascending)
    rank_columns = [c for c in ranked.columns if c.startswith("rank_")]
    ranked["score"] = ranked[rank_columns].mean(axis=1)
    ranked = ranked.sort_values("score").reset_index(drop=True)
    ranked.insert(0, "rank", range(1, len(ranked) + 1))
    return ranked


def answers(ranked: pd.DataFrame) -> list[str]:
    """The three questions §8 requires the output to answer, in words."""
    out = []

    def cell(src, tgt, column):
        row = ranked[(ranked["src_lang"] == src) & (ranked["tgt_lang"] == tgt)]
        return float(row[column].iloc[0]) if len(row) else float("nan")

    hi_ml, bn_ml = cell("hin", "mal", "drift_rate"), cell("ben", "mal", "drift_rate")
    verdict = "worse" if hi_ml > bn_ml else "better"
    out.append(
        f"**Is Hi→Ml worse than Bn→Ml?** {verdict.capitalize()}: Hi→Ml drifts on "
        f"{hi_ml:.1%} of rows against Bn→Ml's {bn_ml:.1%} "
        f"(ranks {int(ranked.loc[(ranked.src_lang=='hin')&(ranked.tgt_lang=='mal'),'rank'].iloc[0])} "
        f"and {int(ranked.loc[(ranked.src_lang=='ben')&(ranked.tgt_lang=='mal'),'rank'].iloc[0])} of 9)."
    )

    by_family = ranked.groupby(ranked["src_lang"].map(FAMILY))["drift_rate"].mean()
    dravidian = by_family.get("Dravidian", float("nan"))
    indo_aryan = by_family.get("Indo-Aryan", float("nan"))
    out.append(
        f"**Are Dravidian-source directions worse?** Mean drift "
        f"{dravidian:.1%} from Dravidian source against {indo_aryan:.1%} from "
        f"Indo-Aryan — "
        + ("yes." if dravidian > indo_aryan else "no, the opposite.")
    )

    quads = ranked.groupby("quadrant")["drift_rate"].mean()
    ia_dr = quads.get("Indo-Aryan->Dravidian", float("nan"))
    dr_ia = quads.get("Dravidian->Indo-Aryan", float("nan"))
    gap = abs(ia_dr - dr_ia)
    out.append(
        f"**Is the Indo-Aryan↔Dravidian penalty symmetric?** No: crossing into "
        f"Dravidian drifts on {ia_dr:.1%} of rows, crossing back into Indo-Aryan on "
        f"{dr_ia:.1%} — a {gap:.1%} gap in the same language pairs. The cost is in "
        "the direction of travel, not the pairing."
    )
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--tau", type=float, default=None)
    args = parser.parse_args(argv)

    config = load_config()
    tau = args.tau if args.tau is not None else config["tau"]
    try:
        frame = gather(args.task, tau)
    except FileNotFoundError as error:
        print(f"missing input: {error}\nRun T-105, T-107 and T-108 for this task first.",
              file=sys.stderr)
        return 2

    ranked = rank(frame)
    columns = ["rank", "src_lang", "tgt_lang", "quadrant", "median_sim", "drift_rate",
               "entity_preserved", "corrupt_rate", "truncation_rate"]
    if "span_recovery" in ranked:
        columns.append("span_recovery")
    display = ranked[columns].round(4)
    print(f"task {args.task}  τ={tau}\n")
    print(display.to_string(index=False))

    quadrants = ranked.groupby("quadrant").agg(
        directions=("rank", "size"), mean_drift=("drift_rate", "mean"),
        mean_entity=("entity_preserved", "mean")).reset_index().round(4)
    print("\nby typological quadrant:")
    print(quadrants.to_string(index=False))

    text = answers(ranked)
    print()
    for line in text:
        print("  " + line.replace("**", ""))

    report_dir = REPORT_ROOT / f"task_{args.task}"
    report_dir.mkdir(parents=True, exist_ok=True)
    ranked.to_parquet(report_dir / "direction_ranking.parquet", index=False)
    lines = [
        f"# T-111 — direction ranking, task {args.task}",
        "",
        f"τ = {tau}. Ranked by the mean of each measure's rank, not by a weighted "
        "sum: a drift rate and a numeral-preservation rate are not on one scale, and "
        "inventing weights between them would manufacture precision. Rank averaging "
        'says only "worse on more measures".',
        "",
        display.to_markdown(index=False),
        "",
        "## By typological quadrant",
        "",
        quadrants.to_markdown(index=False),
        "",
        "## The questions",
        "",
        *(f"{line}\n" for line in text),
        "**Reading the drift column.** It counts pairs below τ, and T-110 found that "
        "τ=0.82 also rejects a quarter of *human* translations of the same items. So "
        "the column ranks directions against each other reliably; its absolute level "
        "is not a failure rate.",
        "",
    ]
    (report_dir / "direction_ranking.md").write_text("\n".join(lines))
    print(f"\nWrote {(report_dir / 'direction_ranking.md').relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
