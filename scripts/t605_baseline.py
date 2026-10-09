"""Saliency divergence on items the model got RIGHT in both languages — the null
distribution the terminology-gap threshold is set against.

    python -m scripts.t605_baseline --task 2 --encoder indicbert-v2 [--per-condition 20] [--seed 0]
    python -m scripts.t605_baseline --report-only        # CPU: rebuild the report from saved rows

Why: a threshold read off the failures' own divergence distribution (say its
95th percentile) fixes the firing rate by construction, which is tuning the label
share (CLAUDE4.md hard rule 3). The principled reference is what divergence looks
like when nothing went wrong. Its 95th percentile is the value a correct pair
exceeds 5% of the time, so a failure above it is unusual for a non-failure:
a ~5% false-positive rate stated in advance, not a share chosen afterwards.

Sample: for every transfer condition, `--per-condition` (item, seed) pairs, seeded,
drawn from those with `pred_src == gold == pred_tgt` (the ceiling log joined to
the condition's target log, as T-307 does) whose source carries a lexicon term the
saliency module can evaluate. The LaBSE gate is not applied: these are correct
predictions, and the gate decides attribution, not eligibility for a null.
Each row is attributed with the same checkpoint, IG baseline and step count
(`saliency.n_steps_for(encoder, task)`) the diagnostics use.

Writes `data/diagnostics/baseline/task_{n}/[{encoder}/]{condition}.parquet`
(resumable: a condition with a file is skipped) and `reports/saliency_baseline.md`.
The report gives the candidate threshold; choosing it is still a documented human
decision, applied with `scripts.t601_rethreshold`.
"""

from __future__ import annotations

import argparse
import random

import pandas as pd

from scripts.t601_diagnostics import resolve_device
from src import saliency
from src.corpus_io import read_split
from src.data import DEFAULT_ENCODER, get_tokenizer
from src.diagnostics import (
    AUDIT_ROOT,
    DIAGNOSTICS_ROOT,
    _free_gpu,
    discover_conditions,
    load_esg_terms,
    path_encoder,
)
from src.download_dataset.paths import REPO_ROOT
from src.evaluate import _split_lang, load_matrix
from src.inference import load_frozen_model
from src.mismatch import join_source_target
from src.predictions import read_prediction_log

BASELINE_ROOT = DIAGNOSTICS_ROOT / "baseline"
REPORT = REPO_ROOT / "reports" / "saliency_baseline.md"
NULL_QUANTILE = 0.95
COLUMNS = ["condition_id", "encoder_id", "item_id", "seed", "block_id", "src_lang", "tgt_lang",
           "concept", "saliency_status", "saliency_divergence", "saliency_convergence_error",
           "saliency_n_steps"]


def baseline_path(task: int, condition_id: str, encoder: str) -> "Path":  # noqa: F821
    base = BASELINE_ROOT / f"task_{task}"
    enc = path_encoder(encoder)
    return (base / enc if enc else base) / f"{condition_id}.parquet"


def both_correct(task: int, condition_id: str, encoder: str, block_src: str, lang_src: str) -> pd.DataFrame:
    enc = path_encoder(encoder)
    source = read_prediction_log(task, f"{block_src}_{lang_src}_native_ceiling", encoder=enc)
    target = read_prediction_log(task, condition_id, encoder=enc)
    joined = join_source_target(source, target)
    keep = (joined["pred_src"] == joined["gold"]) & (joined["pred_tgt"] == joined["gold"])
    return joined[keep]


def sample_condition(task, condition_id, encoder, per_condition, seed, concepts):
    condition = {c["name"]: c for c in load_matrix(task)["conditions"]}[condition_id]
    block_src, lang_src = _split_lang(condition["train"])
    pool = both_correct(task, condition_id, encoder, block_src, lang_src)
    if pool.empty:
        return [], (None, lang_src, None), {}, {}, 0
    block_tgt, lang_tgt = str(pool["block_id"].iloc[0]), str(pool["lang"].iloc[0])
    src = read_split(task, block_src, lang_src, frozen=True)
    tgt = read_split(task, block_tgt, lang_tgt, frozen=True)
    src_text, tgt_text = dict(zip(src["item_id"], src["text"])), dict(zip(tgt["item_id"], tgt["text"]))
    eligible = []
    for r in pool.sort_values(["item_id", "seed"]).itertuples():
        found = saliency.find_lexicon_term(str(src_text[r.item_id]), concepts, lang_src)
        if found and found[1].get(lang_tgt) and found[1][lang_tgt] in str(tgt_text[r.item_id]):
            eligible.append(r)
    rng = random.Random(f"{seed}:{task}:{encoder}:{condition_id}")
    picked = rng.sample(eligible, min(per_condition, len(eligible)))
    return picked, (block_tgt, lang_src, lang_tgt), src_text, tgt_text, len(eligible)


def run(task: int, encoder: str, per_condition: int, seed: int, device: str) -> None:
    tokenizer = get_tokenizer(encoder)
    concepts = load_esg_terms()
    n_steps = saliency.n_steps_for(encoder, task)
    for entry in discover_conditions(task, encoder=path_encoder(encoder)):
        cond = entry["condition_id"]
        out = baseline_path(task, cond, encoder)
        if out.exists():
            print(f"{cond}: done, skipped")
            continue
        picked, (block_tgt, lang_src, lang_tgt), src_text, tgt_text, n_eligible = sample_condition(
            task, cond, encoder, per_condition, seed, concepts)
        rows, loaded, loaded_run = [], None, None
        for r in sorted(picked, key=lambda r: r.tgt_run_id):  # one classifier resident at a time
            if r.tgt_run_id != loaded_run:
                loaded = None
                _free_gpu()
                loaded, loaded_run = load_frozen_model(r.tgt_run_id, device=device), r.tgt_run_id
            res = saliency.diagnose_instance(
                loaded, tokenizer, str(src_text[r.item_id]), str(tgt_text[r.item_id]), int(r.gold),
                concepts, lang_src, lang_tgt, max_len=loaded.run_config.max_len,
                n_steps=n_steps, device=device,
            )
            rows.append({"condition_id": cond, "encoder_id": encoder, "item_id": r.item_id, "seed": int(r.seed),
                         "block_id": block_tgt, "src_lang": lang_src, "tgt_lang": lang_tgt,
                         "concept": res.concept_id, "saliency_status": res.status,
                         "saliency_divergence": res.divergence,
                         "saliency_convergence_error": res.convergence_delta, "saliency_n_steps": n_steps})
        loaded = None
        _free_gpu()
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows, columns=COLUMNS).to_parquet(out, index=False)
        print(f"{cond}: {len(rows)} of {n_eligible} eligible both-correct pairs -> {out}")


def _failures_divergence(task: int, encoder: str) -> pd.Series:
    enc = path_encoder(encoder)
    base = AUDIT_ROOT / f"task_{task}"
    base = base / enc if enc else base
    frames = [pd.read_parquet(p) for p in sorted(base.glob("*.parquet"))]
    if not frames:
        return pd.Series(dtype=float)
    a = pd.concat(frames)
    return a.loc[a["saliency_status"].isin(["fired", "not_fired"]), "saliency_divergence"].dropna()


def report(tasks, encoders) -> None:
    qs = (0.5, 0.9, 0.95, 0.99)
    lines = ["# Saliency divergence — null distribution (both-correct pairs)", "",
             "Generated by `python -m scripts.t605_baseline`. Divergence = the lexicon term's salience share "
             "in the source minus its share in the target. Only converged attributions are counted.", "",
             f"**Candidate threshold** = the null's {NULL_QUANTILE:.0%} quantile: a correctly handled pair "
             "exceeds it ~5% of the time. Choosing it is a one-off decision to record in CLAUDE4.md "
             "(hard rule 3), then apply with `python -m scripts.t601_rethreshold --saliency-threshold X` "
             "and set `saliency.DEFAULT_DIVERGENCE_THRESHOLD` to match.", "",
             "| task | encoder | null n (converged / sampled) | " + " | ".join(f"null q{int(q*100)}" for q in qs)
             + " | failures n | failures ≥ candidate |",
             "|---|---|---|" + "---|" * len(qs) + "---|---|"]
    for task in tasks:
        for encoder in encoders:
            files = sorted(baseline_path(task, "x", encoder).parent.glob("*.parquet"))
            if not files:
                lines.append(f"| {task} | {encoder} | not run | " + " | " * len(qs) + " | |")
                continue
            b = pd.concat([pd.read_parquet(f) for f in files])
            conv = b.loc[b["saliency_status"].isin(["fired", "not_fired"]), "saliency_divergence"].dropna()
            fail = _failures_divergence(task, encoder)
            if conv.empty:
                lines.append(f"| {task} | {encoder} | 0 / {len(b)} | " + " | " * len(qs) + " | |")
                continue
            cand = conv.quantile(NULL_QUANTILE)
            share = (fail >= cand).mean() if len(fail) else float("nan")
            lines.append(f"| {task} | {encoder} | {len(conv)} / {len(b)} | "
                         + " | ".join(f"{conv.quantile(q):+.3f}" for q in qs)
                         + f" | {len(fail)} | {share:.1%} |")
    lines += ["", "If failures sit above the candidate at roughly the null's own ~5%, divergence does not "
              "separate failures from correct pairs, and the honest outcome is that the terminology-gap "
              "module detects nothing beyond chance, reported as such rather than rescued by moving the "
              "threshold.", ""]
    REPORT.write_text("\n".join(lines))
    print(f"wrote {REPORT}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", type=int, action="append")
    parser.add_argument("--encoder", action="append")
    parser.add_argument("--per-condition", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--report-only", action="store_true")
    args = parser.parse_args(argv)
    tasks = args.task or [2, 3]
    encoders = args.encoder or [DEFAULT_ENCODER, "mbert-base"]
    if not args.report_only:
        device = resolve_device(args.device)
        for task in tasks:
            for encoder in encoders:
                print(f"=== task_{task} / {encoder}")
                run(task, encoder, args.per_condition, args.seed, device)
    report(tasks, encoders)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
