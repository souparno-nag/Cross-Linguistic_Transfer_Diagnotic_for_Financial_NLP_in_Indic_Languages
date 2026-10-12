"""Stage 3 — data and model-behaviour attribution over saved diagnostics.

    python -m scripts.t609_behaviour [--task 2 --task 3] [--encoder indicbert-v2 --encoder mbert-base] [--dry-run]

Runs the four `src.behaviour` modules (provenance_specific, cross_lingual_hard,
class_collapse, seed_unstable) on every gate-passed row of a cell, then
re-resolves `modules_fired`, `assigned_label` and `label_group`. A cell is one
(task, encoder): `cross_lingual_hard` needs every condition of it at once,
which is why this runs after `t601_diagnostics` rather than inside it.

Stage 1/2 statuses are left exactly as saved. Idempotent: Stage 3 modules are
stripped from `modules_fired` before being re-derived, so running twice equals
running once. CPU only; reads the prediction logs, no model.
Writes a per-cell summary to `data/diagnostics/behaviour.json`.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src import behaviour
from src.data import DEFAULT_ENCODER
from src.diagnostics import (
    DIAGNOSTICS_ROOT,
    PRECEDENCE,
    audit_path,
    diagnostics_path,
    label_group,
    path_encoder,
    precedence_pick,
    write_diagnostics,
)
from src.evaluate import _split_lang, load_matrix
from src.predictions import read_prediction_log

ORDER = [m for m, _ in PRECEDENCE]


def resolve(audit: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Re-derive `modules_fired` / `assigned_label` from the saved Stage 1/2
    modules plus the Stage 3 statuses now on `audit`."""
    audit, labels = audit.copy(), labels.copy()
    modules, assigned = [], []
    for pos, old in enumerate(audit["modules_fired"]):
        kept = {m for m in list(old) if m not in behaviour.MODULES}
        kept |= {m for m in behaviour.MODULES if audit[f"{m}_status"].iloc[pos] == "fired"}
        ordered = [m for m in ORDER if m in kept]
        modules.append(ordered)
        assigned.append(precedence_pick(ordered))
    audit["modules_fired"], audit["assigned_label"] = modules, assigned
    audit["label_group"] = [label_group(x) for x in assigned]
    by_key = dict(zip(zip(audit["item_id"], audit["seed"]), zip(modules, assigned)))
    keys = list(zip(labels["item_id"], labels["seed"]))
    labels["modules_fired"] = [by_key[k][0] for k in keys]
    labels["assigned_label"] = [by_key[k][1] for k in keys]
    labels["label_group"] = [label_group(x) for x in labels["assigned_label"]]
    return audit, labels


def _log_or_none(task: int, condition_id: str, enc: str | None) -> pd.DataFrame | None:
    try:
        return read_prediction_log(task, condition_id, encoder=enc)
    except FileNotFoundError:
        return None


def run_cell(task: int, encoder: str, *, dry_run: bool = False) -> dict:
    enc = path_encoder(encoder)
    matrix = {c["name"]: c for c in load_matrix(task)["conditions"]}
    paths = sorted(audit_path(task, "x", encoder=enc).parent.glob("*.parquet"))
    audits = {p.stem: pd.read_parquet(p) for p in paths}
    audits = {c: a for c, a in audits.items() if not a.empty}
    failed = behaviour.failed_languages(audits)
    logs: dict = {}

    def log(name):
        if name not in logs:
            logs[name] = _log_or_none(task, name, enc)
        return logs[name]

    cell: dict = {"rows": 0, "labels": {}, "groups": {}}
    for cond, audit in audits.items():
        block_src, lang_src = _split_lang(matrix[cond]["train"])
        annotated = behaviour.annotate(
            audit, cond,
            target_log=log(cond),
            source_log=log(f"{block_src}_{lang_src}_native_ceiling"),
            twin_log=log(behaviour.twin_condition(cond)),
            failed_langs=failed,
        )
        labels = pd.read_parquet(diagnostics_path(task, cond, encoder=enc))
        new_audit, new_labels = resolve(annotated, labels)
        if not dry_run:
            write_diagnostics(task, cond, new_labels, new_audit, encoder=enc)
        cell["rows"] += len(new_audit)
        for col, key in (("assigned_label", "labels"), ("label_group", "groups")):
            for value, n in new_audit[col].value_counts().items():
                cell[key][value] = cell[key].get(value, 0) + int(n)
    return cell


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", type=int, action="append")
    parser.add_argument("--encoder", action="append")
    parser.add_argument("--dry-run", action="store_true", help="compute and print; write nothing")
    args = parser.parse_args(argv)
    tasks = args.task or [2, 3]
    encoders = args.encoder or [DEFAULT_ENCODER, "mbert-base"]

    out: dict = {"cutoffs": {"collapse_share": behaviour.COLLAPSE_SHARE,
                             "xling_min_langs": behaviour.XLING_MIN_LANGS}, "cells": {}}
    for task in tasks:
        for encoder in encoders:
            cell = run_cell(task, encoder, dry_run=args.dry_run)
            out["cells"][f"task_{task}/{encoder}"] = cell
            n = cell["rows"] or 1
            groups = ", ".join(f"{g} {c / n:.1%}" for g, c in sorted(cell["groups"].items()))
            print(f"task_{task} / {encoder}: {cell['rows']} rows; {groups}")
    if not args.dry_run:
        path = DIAGNOSTICS_ROOT / "behaviour.json"
        path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
