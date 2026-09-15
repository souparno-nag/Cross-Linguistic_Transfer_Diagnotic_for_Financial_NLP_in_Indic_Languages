"""Results tables: transfer matrix, quadrant summary, translationese
comparison (CLAUDE3.md T-308).

Computes nothing new -- every number here is T-304's bootstrap CI and
T-306's paired (item_id, seed) join, aggregated over the 3 seeds hard rule 5
requires and rendered for the Review 3 deck. A condition whose checkpoint
does not exist yet (T-305: Bengali/Telugu baselines are still deferred) is
reported as blocked, not omitted, so the matrix's shape stays the 9 cells
CLAUDE3.md's evaluation-conditions table describes regardless of how many
are currently fillable.

The **source** score for a gap is not T-207's baseline number (computed on a
different, randomly stratified test fold). It is the model's own accuracy on
exactly the items paired with the target arm via T-306's join -- the same
items, scored in the training language -- so `source - target` compares the
model against itself on identical content, which is what "transfer gap"
means (Goal, CLAUDE3.md).
"""

from __future__ import annotations

import statistics

import pandas as pd

from .data import num_labels
from .evaluate import _split_lang, load_matrix
from .metrics import classification_metrics
from .mismatch import join_source_target
from .data import DEFAULT_ENCODER
from .predictions import read_prediction_log
from .transfer import DEFAULT_N_BOOT, bootstrap_metric, transfer_gap

QUADRANTS = [
    "Indo-Aryan->Indo-Aryan",
    "Indo-Aryan->Dravidian",
    "Dravidian->Indo-Aryan",
    "Dravidian->Dravidian",
]


def _source_id_for(condition: dict) -> str:
    block, lang = _split_lang(condition["train"])
    return f"{block}_{lang}_native_ceiling"


def _path_encoder(encoder: str | None) -> str | None:
    """The encoder level a prediction log lives under, or `None` for the
    default encoder, whose logs keep the original un-namespaced path
    (`predictions.log_path`)."""
    return None if encoder in (None, DEFAULT_ENCODER) else encoder


def condition_seed_gap(
    task: int,
    condition: dict,
    seed: int,
    *,
    n_boot: int = DEFAULT_N_BOOT,
    encoder: str | None = None,
) -> dict | None:
    """One (condition, seed)'s paired gap. `None` if either log, or their
    paired-item overlap, is empty -- the caller reports that as blocked."""
    try:
        enc = _path_encoder(encoder)
        source = read_prediction_log(task, _source_id_for(condition), encoder=enc)
        target = read_prediction_log(task, condition["name"], encoder=enc)
    except FileNotFoundError:
        return None
    source = source[source["seed"] == seed]
    target = target[target["seed"] == seed]
    if source.empty or target.empty:
        return None
    joined = join_source_target(source, target)
    if joined.empty:
        return None

    labels = num_labels(task)
    src_boot = bootstrap_metric(joined["gold"], joined["pred_src"], labels, n_boot=n_boot, seed=seed)
    tgt_boot = bootstrap_metric(joined["gold"], joined["pred_tgt"], labels, n_boot=n_boot, seed=seed)
    gap = transfer_gap(src_boot, tgt_boot)
    return {"n_items": len(joined), **gap.to_dict()}


def condition_summary(
    task: int,
    condition: dict,
    *,
    seeds: tuple[int, ...] = (0, 1, 2),
    n_boot: int = DEFAULT_N_BOOT,
    encoder: str | None = None,
) -> dict:
    """Mean +/- sample-std over seeds (hard rule 3), plus whether the gap is
    smaller than that seed-to-seed spread (hard rule 5: not a finding then)."""
    per_seed = [
        condition_seed_gap(task, condition, s, n_boot=n_boot, encoder=encoder)
        for s in seeds
    ]
    available = [r for r in per_seed if r is not None]
    base = {
        "condition": condition["name"],
        "quadrant": condition.get("quadrant"),
        "n_seeds": len(available),
    }
    if not available:
        return {**base, "status": "blocked"}

    gaps = [r["gap"] for r in available]
    gap_std = statistics.stdev(gaps) if len(gaps) > 1 else 0.0
    gap_mean = statistics.fmean(gaps)
    return {
        **base,
        "status": "ok",
        "n_items": available[0]["n_items"],
        "source_mean": statistics.fmean(r["source"] for r in available),
        "target_mean": statistics.fmean(r["target"] for r in available),
        "gap_mean": gap_mean,
        "gap_std": gap_std,
        "within_seed_noise": abs(gap_mean) < gap_std if len(gaps) > 1 else None,
    }


def condition_matrix(
    task: int, kind: str, *, n_boot: int = DEFAULT_N_BOOT, encoder: str | None = None
) -> pd.DataFrame:
    """Every condition of one `kind` ("transfer" or "transfer_mt"), blocked
    or not -- the shape is always the full 9 cells."""
    matrix = load_matrix(task)
    rows = [
        condition_summary(task, c, n_boot=n_boot, encoder=encoder)
        for c in matrix["conditions"]
        if c["kind"] == kind
    ]
    return pd.DataFrame(rows)


def quadrant_summary(
    task: int,
    kind: str = "transfer",
    *,
    n_boot: int = DEFAULT_N_BOOT,
    encoder: str | None = None,
) -> pd.DataFrame:
    """Mean gap per typological quadrant, over whichever cells are runnable."""
    df = condition_matrix(task, kind, n_boot=n_boot, encoder=encoder)
    ok = df[df["status"] == "ok"] if not df.empty else df
    rows = []
    for quadrant in QUADRANTS:
        sub = ok[ok["quadrant"] == quadrant] if not ok.empty else ok
        if sub.empty:
            rows.append({"quadrant": quadrant, "n_cells": 0, "gap_mean": None, "cells": []})
            continue
        rows.append(
            {
                "quadrant": quadrant,
                "n_cells": len(sub),
                "gap_mean": float(sub["gap_mean"].mean()),
                "cells": sub["condition"].tolist(),
            }
        )
    return pd.DataFrame(rows)


def translationese_comparison(
    task: int, lang: str, *, seeds: tuple[int, ...] = (0, 1, 2), encoder: str | None = None
) -> pd.DataFrame:
    """Per-arm macro-F1, mean +/- std over seeds, and the delta from the
    native arm -- isolates the effect of translation provenance alone,
    holding language and label constant (T-113 §8)."""
    try:
        log = read_prediction_log(
            task, f"translationese_{lang}", encoder=_path_encoder(encoder)
        )
    except FileNotFoundError:
        return pd.DataFrame()

    labels = num_labels(task)
    rows = []
    for (block, origin), group in log.groupby(["block_id", "origin"]):
        per_seed_f1 = []
        for seed in seeds:
            sub = group[group["seed"] == seed]
            if sub.empty:
                continue
            per_seed_f1.append(classification_metrics(sub["gold"], sub["pred"], labels)["macro_f1"])
        if not per_seed_f1:
            continue
        rows.append(
            {
                "block": block,
                "origin": origin,
                "src_lang": (group["src_lang"].dropna().iloc[0] if origin == "mt" else None),
                "n_seeds": len(per_seed_f1),
                "macro_f1_mean": statistics.fmean(per_seed_f1),
                "macro_f1_std": statistics.stdev(per_seed_f1) if len(per_seed_f1) > 1 else 0.0,
            }
        )
    frame = pd.DataFrame(rows)
    if not frame.empty and (frame["origin"] == "native").any():
        native_mean = float(frame.loc[frame["origin"] == "native", "macro_f1_mean"].iloc[0])
        frame["delta_vs_native"] = frame["macro_f1_mean"] - native_mean
    return frame


# --------------------------------------------------------------------------
# Markdown rendering (T-308's actual deliverable table format)
# --------------------------------------------------------------------------


def is_missing(x) -> bool:
    """`None` or NaN. Needed because pandas silently turns an all-`None`
    column into NaN (e.g. every row's `within_seed_noise` is `None` when no
    condition has more than one seed yet), and `bool(float("nan"))` is
    `True` in Python -- an `is None` check alone would misrender a
    genuinely-missing value as truthy."""
    return x is None or (isinstance(x, float) and x != x)


def fmt(x, digits: int = 4) -> str:
    return "—" if is_missing(x) else f"{x:.{digits}f}"


def render_matrix_table(df: pd.DataFrame, *, title: str) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| condition | quadrant | status | n seeds | source | target | gap (mean±std) | within seed noise? |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, row in df.iterrows():
        if row["status"] == "blocked":
            lines.append(f"| {row['condition']} | {row['quadrant']} | blocked | 0 | — | — | — | — |")
            continue
        gap = f"{fmt(row['gap_mean'])} ± {fmt(row['gap_std'])}"
        noise = "—" if is_missing(row["within_seed_noise"]) else ("yes" if row["within_seed_noise"] else "no")
        lines.append(
            f"| {row['condition']} | {row['quadrant']} | ok | {row['n_seeds']} | "
            f"{fmt(row['source_mean'])} | {fmt(row['target_mean'])} | {gap} | {noise} |"
        )
    lines.append("")
    return lines


def render_quadrant_table(df: pd.DataFrame) -> list[str]:
    lines = ["### Quadrant summary", "", "| quadrant | n cells (of 3) | mean gap | cells |", "|---|---|---|---|"]
    for _, row in df.iterrows():
        cells = ", ".join(row["cells"]) if row["cells"] else "—"
        lines.append(f"| {row['quadrant']} | {row['n_cells']} | {fmt(row['gap_mean'])} | {cells} |")
    lines.append("")
    return lines


def render_translationese_table(df: pd.DataFrame, lang: str) -> list[str]:
    lines = [f"### Translationese comparison — {lang}", ""]
    if df.empty:
        lines += [f"_no `translationese_{lang}` prediction log on disk yet._", ""]
        return lines
    lines += [
        "| block | origin | src_lang | n seeds | macro-F1 (mean±std) | Δ vs native |",
        "|---|---|---|---|---|---|",
    ]
    for _, row in df.iterrows():
        delta = fmt(row.get("delta_vs_native")) if "delta_vs_native" in df.columns else "—"
        lines.append(
            f"| {row['block']} | {row['origin']} | {row['src_lang'] or '—'} | {row['n_seeds']} | "
            f"{fmt(row['macro_f1_mean'])} ± {fmt(row['macro_f1_std'])} | {delta} |"
        )
    lines.append("")
    return lines
