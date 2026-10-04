"""Summarise a Phase 6 run, and export the pairs for inspecting the saliency threshold.

    python -m scripts.t601_report [--task 2 --task 3] [--encoder indicbert-v2 --encoder mbert-base]

CPU only; reads `data/diagnostics/`. Writes `reports/diagnostics_summary.md`:

* per (task, encoder, condition): instances, label counts, the unattributed rate
  (CLAUDE4.md: the honest measure of coverage, reported rather than buried) and how
  often each module returned each status — including `not_converged` and
  `unavailable`, which must not be read as "did not fire";
* the distribution of `saliency_divergence`, and the 30 evaluated instances
  nearest the threshold on each side, with their full text, for the one-off manual
  inspection that sets `saliency.DEFAULT_DIVERGENCE_THRESHOLD` (hard rule 3: set once,
  on evidence, documented; never tuned afterwards to tidy the label shares).

No labels are interpreted here — that is Phase 7.
"""

from __future__ import annotations

import argparse

import pandas as pd

from src import saliency
from src.corpus_io import read_split
from src.data import DEFAULT_ENCODER
from src.diagnostics import AUDIT_ROOT, DIAGNOSTICS_ROOT, path_encoder
from src.download_dataset.paths import REPO_ROOT
from src.evaluate import _split_lang, load_matrix

PER_SIDE = 30


def _cell(text) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def _md(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False)


def load_audits(task: int, encoder: str) -> pd.DataFrame:
    enc = path_encoder(encoder)
    base = AUDIT_ROOT / f"task_{task}"
    base = base / enc if enc else base
    frames = [pd.read_parquet(p).assign(task=task) for p in sorted(base.glob("*.parquet"))]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def condition_table(audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cond, g in audit.groupby("condition_id"):
        n = len(g)
        row = {"condition": cond, "n": n,
               "unattributed_%": round(100 * (g["assigned_label"] == "unattributed").mean(), 1)}
        for label, share in g["assigned_label"].value_counts().items():
            row[label] = int(share)
        for col in ("frag_status", "morph_status", "saliency_status"):
            for status, count in g[col].value_counts().items():
                row[f"{col.split('_')[0]}:{status}"] = int(count)
        rows.append(row)
    return pd.DataFrame(rows).fillna(0)


def nearest_to_threshold(task: int, audit: pd.DataFrame, threshold: float) -> pd.DataFrame:
    ev = audit[audit["saliency_status"].isin(["fired", "not_fired"])].copy()
    if ev.empty:
        return ev
    ev["gap"] = ev["saliency_divergence"] - threshold
    below = ev[ev["gap"] < 0].nlargest(PER_SIDE, "gap")
    above = ev[ev["gap"] >= 0].nsmallest(PER_SIDE, "gap")
    pick = pd.concat([below, above])
    matrix = {c["name"]: c for c in load_matrix(task)["conditions"]}
    texts: dict = {}

    def text(block, lang, item):
        key = (block, lang)
        if key not in texts:
            frame = read_split(task, block, lang, frozen=True)
            texts[key] = dict(zip(frame["item_id"], frame["text"]))
        return texts[key].get(item, "")

    out = []
    for _, r in pick.iterrows():
        sb, sl = _split_lang(matrix[r["condition_id"]]["train"])
        out.append({
            "condition": r["condition_id"], "item_id": r["item_id"],
            "divergence": round(float(r["saliency_divergence"]), 3),
            "conv_err": round(float(r["saliency_convergence_error"]), 4)
            if "saliency_convergence_error" in r and pd.notna(r["saliency_convergence_error"]) else None,
            "status": r["saliency_status"],
            "src": _cell(text(sb, sl, r["item_id"])),
            "tgt": _cell(text(r["block_id"], r["tgt_lang"], r["item_id"])),
        })
    return pd.DataFrame(out)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", type=int, action="append")
    parser.add_argument("--encoder", action="append")
    args = parser.parse_args(argv)
    tasks = args.task or [2, 3]
    encoders = args.encoder or [DEFAULT_ENCODER, "mbert-base"]

    out = ["# Phase 6 diagnostics — run summary", "",
           f"Saliency divergence threshold in force: **{saliency.DEFAULT_DIVERGENCE_THRESHOLD}** "
           f"(placeholder until inspected below); convergence tolerance "
           f"{saliency.DEFAULT_CONVERGENCE_TOL:.0%} relative.", ""]
    for encoder in encoders:
        for task in tasks:
            audit = load_audits(task, encoder)
            out += [f"## task {task} — {encoder}", ""]
            if audit.empty:
                out += ["_no diagnostics on disk yet._", ""]
                continue
            n = len(audit)
            un = (audit["assigned_label"] == "unattributed").mean()
            out += [f"{n} instances; **unattributed {un:.1%}**.", "", _md(condition_table(audit)), ""]
            div = audit.loc[audit["saliency_status"].isin(["fired", "not_fired"]), "saliency_divergence"].dropna()
            if len(div):
                out += ["Saliency divergence over converged, evaluated instances "
                        f"(n={len(div)}): " + ", ".join(
                            f"{q:.0%}={div.quantile(q):+.3f}" for q in (0.05, 0.25, 0.5, 0.75, 0.95)),
                        ""]
                near = nearest_to_threshold(task, audit, saliency.DEFAULT_DIVERGENCE_THRESHOLD)
                parquet = REPO_ROOT / "reports" / f"task_{task}" / f"saliency_threshold_pairs_{encoder}.parquet"
                parquet.parent.mkdir(parents=True, exist_ok=True)
                near.to_parquet(parquet, index=False)
                out += [f"### {PER_SIDE} instances either side of the threshold ({parquet.name})", "", _md(near), ""]
            else:
                out += ["No instance reached the saliency module's verdict (all `not_applicable`, "
                        "`not_converged` or `unavailable`).", ""]
    path = REPO_ROOT / "reports" / "diagnostics_summary.md"
    path.write_text("\n".join(out) + "\n")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
