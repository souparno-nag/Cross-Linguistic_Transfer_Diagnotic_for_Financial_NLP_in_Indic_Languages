"""Re-apply a saliency-divergence threshold to saved diagnostics, offline.

    python -m scripts.t601_rethreshold --saliency-threshold 0.05 [--task 2 --task 3] [--encoder ...]

Changing `saliency.DEFAULT_DIVERGENCE_THRESHOLD` after the T-605 inspection should not
cost a GPU re-run: every audit row stores the divergence it was judged on. This re-derives
`saliency_status`, `modules_fired` and `assigned_label` from the stored divergence and
leaves everything else untouched. CPU only, deterministic, idempotent (running twice with
one value equals running once).

What it does NOT touch: gate-terminal rows, `not_converged` / `not_applicable` /
`unavailable` rows, and rows with no convergence record (written before the convergence
check existed) — those are counted and reported, never silently re-judged.

It does not choose the threshold. That is a one-off decision made on the inspected pairs
and documented (CLAUDE4.md hard rule 3); also set `saliency.DEFAULT_DIVERGENCE_THRESHOLD`
to the same value, or the next `t601_diagnostics --force` will reproduce the old labels.
The value in force is recorded in `data/diagnostics/thresholds.json`.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src import saliency
from src.data import DEFAULT_ENCODER
from src.diagnostics import (
    DIAGNOSTICS_ROOT,
    PRECEDENCE,
    audit_path,
    diagnostics_path,
    path_encoder,
    precedence_pick,
    write_diagnostics,
)

ORDER = [m for m, _ in PRECEDENCE]
JUDGED = ["fired", "not_fired"]


def rethreshold(audit: pd.DataFrame, labels: pd.DataFrame, threshold: float):
    """Returns `(audit, labels, stats)` with the saliency verdict recomputed."""
    audit = audit.copy()
    labels = labels.copy()
    if "saliency_convergence_error" not in audit:
        audit["saliency_convergence_error"] = float("nan")
    judged = audit["saliency_status"].isin(JUDGED) & audit["saliency_divergence"].notna()
    eligible = judged & audit["saliency_convergence_error"].notna()

    new_status = audit["saliency_status"].copy()
    new_status[eligible] = audit.loc[eligible, "saliency_divergence"].map(
        lambda d: "fired" if saliency.fires(float(d), threshold) else "not_fired"
    )

    modules, assigned = [], []
    for old, status in zip(audit["modules_fired"], new_status):
        kept = {m for m in list(old) if m != "saliency"}
        if status == "fired":
            kept.add("saliency")
        ordered = [m for m in ORDER if m in kept]
        modules.append(ordered)
        assigned.append(precedence_pick(ordered))

    stats = {
        "eligible": int(eligible.sum()),
        "no_convergence_record": int((judged & ~eligible).sum()),
        "status_changed": int((new_status != audit["saliency_status"]).sum()),
        "label_changed": int((pd.Series(assigned, index=audit.index) != audit["assigned_label"]).sum()),
        "saliency_fired": int((new_status == "fired").sum()),
    }
    audit["saliency_status"] = new_status
    audit["modules_fired"] = modules
    audit["assigned_label"] = assigned

    key = list(zip(audit["item_id"], audit["seed"]))
    by_key = dict(zip(key, zip(modules, assigned)))
    lkey = list(zip(labels["item_id"], labels["seed"]))
    labels["modules_fired"] = [by_key[k][0] for k in lkey]
    labels["assigned_label"] = [by_key[k][1] for k in lkey]
    return audit, labels, stats


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--saliency-threshold", type=float, required=True)
    parser.add_argument("--task", type=int, action="append")
    parser.add_argument("--encoder", action="append")
    args = parser.parse_args(argv)
    tasks = args.task or [2, 3]
    encoders = args.encoder or [DEFAULT_ENCODER, "mbert-base"]

    log: dict = {"saliency_threshold": args.saliency_threshold, "conditions": {}}
    for encoder in encoders:
        enc = path_encoder(encoder)
        for task in tasks:
            base = audit_path(task, "x", encoder=enc).parent
            for path in sorted(base.glob("*.parquet")):
                cond = path.stem
                audit = pd.read_parquet(audit_path(task, cond, encoder=enc))
                labels = pd.read_parquet(diagnostics_path(task, cond, encoder=enc))
                new_audit, new_labels, stats = rethreshold(audit, labels, args.saliency_threshold)
                write_diagnostics(task, cond, new_labels, new_audit, encoder=enc)
                log["conditions"][f"task_{task}/{encoder}/{cond}"] = stats
                print(f"task_{task} {encoder} {cond}: {stats}")
    out = DIAGNOSTICS_ROOT / "thresholds.json"
    out.write_text(json.dumps(log, indent=2, sort_keys=True) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
