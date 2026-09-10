"""T-108 — LaBSE similarity for every MT pair.

    python -m scripts.t108_labse --task 3               # ~1 min
    python -m scripts.t108_labse --task 2               # ~4 min
    python -m scripts.t108_labse --task 1               # ~15 min
    python -m scripts.t108_labse --task 1 --scores-only # reuse cached vectors

Embeds every native and MT split once, scores every MT row against the source
row it came from, and writes `labse_sim` into the corpus. Also scores MT
against the human reference translation where one exists (tasks 2 and 3,
hin/ben/tel only) — that is T-110's quality ceiling.

Needs the GPU. Embeddings are cached under `data/verification/task_{n}/emb/`,
so a re-run after an interruption re-embeds only what is missing, and
`--scores-only` recomputes the scores from cache without loading the model at
all.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import pandas as pd

from src.download_dataset.paths import REPO_ROOT
from src.labse_gate import (
    VERIFICATION_ROOT,
    Embedder,
    apply_scores,
    load_config,
    reference_scores,
    tau_for,
    score_task,
    verification_fingerprint,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--scores-only",
        action="store_true",
        help="recompute from cached embeddings; loads no model",
    )
    parser.add_argument(
        "--no-write", action="store_true", help="do not write labse_sim into the corpus"
    )
    args = parser.parse_args(argv)

    config = load_config()
    if args.batch_size:
        config["batch_size"] = args.batch_size
    fingerprint = verification_fingerprint(config)
    print(f"model {config['model']}  fingerprint {fingerprint}  tau {config['tau']}")

    embedder = None
    if not args.scores_only:
        embedder = Embedder(config, device=args.device)
        print(f"loaded on {embedder.device}")

    try:
        scores = score_task(embedder, args.task)
        references = reference_scores(embedder, args.task)
    except RuntimeError as error:
        print(f"\n{error}", file=sys.stderr)
        return 1

    tau = tau_for(config, args.task)
    summary = (
        scores.groupby(["src_lang", "tgt_lang"])["labse_sim"]
        .agg(rows="size", mean="mean", median="median", p05=lambda s: s.quantile(0.05))
        .reset_index()
    )
    below = (
        scores.assign(below=scores["labse_sim"] < tau)
        .groupby(["src_lang", "tgt_lang"])["below"]
        .agg(below_tau="sum", below_share="mean")
        .reset_index()
    )
    summary = summary.merge(below, on=["src_lang", "tgt_lang"]).round(4)
    print("\n" + summary.to_string(index=False))

    out_dir = VERIFICATION_ROOT / f"task_{args.task}"
    out_dir.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(out_dir / "labse_scores.parquet", index=False)
    # Rule 8, plus the device: LaBSE runs float32 on both CPU and GPU so the
    # difference is float noise rather than the float16 gap that made T-106's
    # CPU fallback dangerous — but scores from two devices are still not
    # byte-identical, and rule 7 asks that to be visible rather than assumed.
    (out_dir / "labse_run.json").write_text(
        json.dumps(
            {
                "task": args.task,
                "model": config["model"],
                "fingerprint": fingerprint,
                "tau": tau,
                "device": (embedder.device if embedder else "cached embeddings"),
                "pairs": len(scores),
                "reference_pairs": len(references),
                "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nWrote {(out_dir / 'labse_scores.parquet').relative_to(REPO_ROOT)} "
          f"({len(scores)} pairs)")

    if len(references):
        references.to_parquet(out_dir / "reference_scores.parquet", index=False)
        ref_summary = (
            references.groupby(["src_lang", "tgt_lang"])["reference_sim"]
            .agg(rows="size", median="median")
            .reset_index()
            .round(4)
        )
        print(f"\nagainst human reference translations (§2.1):")
        print(ref_summary.to_string(index=False))
    else:
        print(
            "\nNo human reference available for this task — task 1 is independently "
            "sourced, so every direction falls back to source similarity alone (§2.1)."
        )

    if not args.no_write:
        written = apply_scores(args.task, scores)
        print(f"\nlabse_sim written into {len(written)} splits, {sum(written.values())} rows")

    missing = int(scores["labse_sim"].isna().sum())
    if missing:
        print(f"{missing} pairs have no score", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
