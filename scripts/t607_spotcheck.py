"""T-607 (CLAUDE4.md's T-608) — manual spot-check sample for task 2.

    python -m scripts.t607_spotcheck --task 2 --n 50

Samples up to `--n` diagnosed rows per target language (ben/tel/mal — hin
never appears as a target while only Hindi-sourced conditions exist) from
`data/diagnostics/task_2/*.parquet`, pulls the real source/target text and
the deciding evidence from the audit trail, and writes
`reports/task_2/spotcheck.md` with an empty "agree?" column for the project
owner to fill in by hand.

This script cannot compute the agreement rate itself — that needs a human
who can judge whether `assigned_label` is plausible given the evidence and
the actual sentences, which is exactly what CLAUDE4.md's T-608 asks a person
to do. Re-run `scripts.t111_ranking`-style aggregation once the column is
filled in, or just count by hand; the table is small by design.
"""

from __future__ import annotations

import argparse
import random

import pandas as pd

from src.corpus_io import read_split
from src.diagnostics import audit_path, read_diagnostics
from src.download_dataset.paths import REPO_ROOT
from src.evaluate import _split_lang, load_matrix

REPORTS_DIR = REPO_ROOT / "reports"


def _clip(text, n: int = 70) -> str:
    text = str(text)
    return text[:n] + ("…" if len(text) > n else "")


def _evidence_str(row) -> str:
    parts = [f"label={row['assigned_label']}", f"labse_sim={row['labse_sim']:.3f}"]
    if pd.notna(row.get("r_frag")):
        parts.append(f"r_frag={row['r_frag']:.2f}")
    if row.get("morph_status") == "fired" and row.get("morph_evidence"):
        ev = row["morph_evidence"]
        parts.append(f"morph={ev.get('term')}+{ev.get('suffix')}")
    if row.get("saliency_status") == "fired" and pd.notna(row.get("saliency_divergence")):
        parts.append(f"saliency_div={row['saliency_divergence']:.2f}")
    return "; ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--n", type=int, default=50, help="rows per target language")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    matrix = load_matrix(args.task)
    by_name = {c["name"]: c for c in matrix["conditions"]}

    frames = []
    for path in sorted((REPO_ROOT / "data" / "diagnostics" / f"task_{args.task}").glob("*.parquet")):
        condition_id = path.stem
        condition = by_name.get(condition_id)
        if condition is None:
            continue
        block_src, lang_src = _split_lang(condition["train"])
        labels = read_diagnostics(args.task, condition_id)
        audit = pd.read_parquet(audit_path(args.task, condition_id))
        merged = labels.merge(
            audit[["item_id", "seed", "morph_status", "morph_evidence", "saliency_status", "saliency_divergence"]],
            on=["item_id", "seed"], how="left",
        )
        merged["_block_src"] = block_src
        merged["_lang_src"] = lang_src
        frames.append(merged)

    if not frames:
        print("nothing to sample; run scripts.t601_diagnostics first")
        return 0

    pool = pd.concat(frames, ignore_index=True)
    rng = random.Random(args.seed)

    lines = [
        f"# T-607 spot-check — task {args.task}",
        "",
        "Sampled diagnosed rows per target language for manual judgment against the "
        "assigned label. Fill in the **agree?** column by hand (y/n), then report the "
        "agreement rate; if a systematic disagreement shows up, adjust **one** "
        "threshold once, document why, and stop -- do not iterate to a tidy number "
        "(CLAUDE4.md hard rule 3).",
        "",
    ]

    total_sampled = 0
    for lang in sorted(pool["tgt_lang"].unique()):
        subset = pool[pool["tgt_lang"] == lang]
        sample_n = min(args.n, len(subset))
        sample = subset.sample(n=sample_n, random_state=rng.randint(0, 2**31))
        total_sampled += len(sample)
        lines += [
            f"## Target language: {lang} ({len(sample)} of {len(subset)} diagnosed rows)",
            "",
            "| item_id | seed | assigned_label | evidence | src text | tgt text | agree? |",
            "|---|---|---|---|---|---|---|",
        ]
        for _, row in sample.iterrows():
            src_text = read_split(args.task, row["_block_src"], row["_lang_src"], frozen=True)
            src_text = src_text.loc[src_text["item_id"] == row["item_id"], "text"]
            src_text = src_text.iloc[0] if len(src_text) else ""
            tgt_text = read_split(args.task, row["block_id"], row["tgt_lang"], frozen=True)
            tgt_text = tgt_text.loc[tgt_text["item_id"] == row["item_id"], "text"]
            tgt_text = tgt_text.iloc[0] if len(tgt_text) else ""
            lines.append(
                f"| {row['item_id']} | {row['seed']} | {row['assigned_label']} | "
                f"{_evidence_str(row)} | {_clip(src_text)} | {_clip(tgt_text)} | |"
            )
        lines.append("")

    lines += [
        f"**{total_sampled} rows sampled across {pool['tgt_lang'].nunique()} target "
        "language(s). Agreement rate: TBD -- fill in the agree? column and count.**",
    ]

    out_dir = REPORTS_DIR / f"task_{args.task}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "spotcheck.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {out_path} ({total_sampled} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
