"""Re-run Integrated Gradients on saved diagnostics at a higher step count.

    python -m scripts.t605_resaliency --task 3 --encoder indicbert-v2 [--all] [--device cuda]

Recomputes the saliency module only — gate, fragmentation and morphology are
left exactly as saved — for rows whose IG ran at fewer steps than
`saliency.n_steps_for(encoder, task)` now asks for. By default only rows that
came out `not_converged` are redone: a row that already converged has an
attribution accurate to within the tolerance, and more steps would move it
only inside that tolerance. `--all` redoes every IG-evaluated row instead.

Each recomputed row records its own `saliency_n_steps`, so a cell may hold rows
at two step counts, and the audit says which. Labels are then re-resolved by
the same code as `t601_rethreshold`, at the threshold in force.

Resumable: output is written after each condition, and a row already at the
target step count is never redone. Needs a GPU; no silent CPU fallback.
"""

from __future__ import annotations

import argparse

import pandas as pd

from scripts.t601_diagnostics import resolve_device
from scripts.t601_rethreshold import rethreshold
from src import saliency
from src.corpus_io import read_split
from src.data import DEFAULT_ENCODER, get_tokenizer
from src.diagnostics import (
    SALIENCY_EVALUATED,
    _free_gpu,
    audit_path,
    diagnostics_path,
    load_esg_terms,
    path_encoder,
    write_diagnostics,
)
from src.evaluate import _split_lang, load_matrix
from src.failures import read_failures
from src.inference import load_frozen_model


def prior_steps(audit: pd.DataFrame, encoder: str) -> pd.Series:
    """Steps each row was last computed at. Rows from before the column existed
    ran at the per-encoder default of that run (`N_STEPS_BY_ENCODER`)."""
    fallback = saliency.N_STEPS_BY_ENCODER.get(encoder, saliency.DEFAULT_N_STEPS)
    if "saliency_n_steps" not in audit:
        return pd.Series(fallback, index=audit.index)
    return audit["saliency_n_steps"].fillna(fallback).astype(int)


def rows_to_redo(audit: pd.DataFrame, encoder: str, target: int, redo_all: bool) -> pd.Series:
    evaluated = audit["saliency_status"].isin(SALIENCY_EVALUATED)
    if not redo_all:
        evaluated &= audit["saliency_status"] == "not_converged"
    return evaluated & (prior_steps(audit, encoder) < target)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    parser.add_argument("--all", action="store_true", help="redo every IG-evaluated row, not only unconverged ones")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    enc = path_encoder(args.encoder)
    target = saliency.n_steps_for(args.encoder, args.task)
    matrix = {c["name"]: c for c in load_matrix(args.task)["conditions"]}
    paths = sorted(audit_path(args.task, "x", encoder=enc).parent.glob("*.parquet"))
    plan = []
    for path in paths:
        audit = pd.read_parquet(path)
        n = int(rows_to_redo(audit, args.encoder, target, args.all).sum())
        if n:
            plan.append((path.stem, n))
    total = sum(n for _, n in plan)
    print(f"task_{args.task} / {args.encoder}: target {target} steps; {total} rows in {len(plan)} conditions to redo")
    if args.dry_run or not plan:
        for cond, n in plan:
            print(f"  {cond}: {n}")
        return 0

    device = resolve_device(args.device)
    tokenizer = get_tokenizer(args.encoder)
    concepts = load_esg_terms()
    before_after = []
    for cond, _ in plan:
        audit = pd.read_parquet(audit_path(args.task, cond, encoder=enc))
        labels = pd.read_parquet(diagnostics_path(args.task, cond, encoder=enc))
        if "saliency_n_steps" not in audit:
            audit["saliency_n_steps"] = prior_steps(audit, args.encoder).where(
                audit["saliency_status"].isin(SALIENCY_EVALUATED))
        audit["saliency_n_steps"] = audit["saliency_n_steps"].astype("float")
        redo = rows_to_redo(audit, args.encoder, target, args.all)

        failures = read_failures(args.task, cond, encoder=enc)
        by_key = {(r.item_id, int(r.seed)): (int(r.gold), r.tgt_run_id) for r in failures.itertuples()}
        block_src, lang_src = _split_lang(matrix[cond]["train"])
        src = read_split(args.task, block_src, lang_src, frozen=True)
        src_text = dict(zip(src["item_id"], src["text"]))
        first = audit.loc[redo].iloc[0]
        tgt = read_split(args.task, first["block_id"], first["tgt_lang"], frozen=True)
        tgt_text = dict(zip(tgt["item_id"], tgt["text"]))

        idx = list(audit.index[redo])
        idx.sort(key=lambda i: by_key[(audit.at[i, "item_id"], int(audit.at[i, "seed"]))][1])  # one model at a time
        loaded, loaded_run = None, None
        n_before = int((audit["saliency_status"] == "not_converged").sum())
        for i in idx:
            gold, run_id = by_key[(audit.at[i, "item_id"], int(audit.at[i, "seed"]))]
            if run_id != loaded_run:
                loaded = None
                _free_gpu()
                loaded, loaded_run = load_frozen_model(run_id, device=device), run_id
            result = saliency.diagnose_instance(
                loaded, tokenizer, str(src_text[audit.at[i, "item_id"]]), str(tgt_text[audit.at[i, "item_id"]]),
                gold, concepts, lang_src, str(audit.at[i, "tgt_lang"]),
                max_len=loaded.run_config.max_len, n_steps=target, device=device,
            )
            audit.at[i, "saliency_status"] = result.status
            audit.at[i, "saliency_divergence"] = result.divergence
            audit.at[i, "saliency_convergence_error"] = result.convergence_delta
            audit.at[i, "saliency_n_steps"] = target
        loaded = None
        _free_gpu()

        audit, labels, _ = rethreshold(audit, labels, saliency.DEFAULT_DIVERGENCE_THRESHOLD)
        write_diagnostics(args.task, cond, labels, audit, encoder=enc)
        n_after = int((audit["saliency_status"] == "not_converged").sum())
        evaluated = int(audit["saliency_status"].isin(SALIENCY_EVALUATED).sum())
        before_after.append({"condition": cond, "redone": len(idx), "evaluated": evaluated,
                             "not_converged_before": n_before, "not_converged_after": n_after})
        print(f"{cond}: redid {len(idx)}; not_converged {n_before} -> {n_after} of {evaluated}")

    print()
    print(pd.DataFrame(before_after).to_string(index=False))
    cell = pd.concat([pd.read_parquet(p) for p in paths])
    ev = int(cell["saliency_status"].isin(SALIENCY_EVALUATED).sum())
    nc = int((cell["saliency_status"] == "not_converged").sum())
    rate = nc / ev if ev else 0.0
    print(f"\ntask_{args.task} / {args.encoder}: {nc}/{ev} IG-evaluated rows unconverged ({rate:.1%}); "
          f"limit {saliency.MAX_UNCONVERGED_RATE:.0%} -> {'within' if rate <= saliency.MAX_UNCONVERGED_RATE else 'ABOVE'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
