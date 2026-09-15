"""T-307 — export the failed-instance set, per condition, plus a sanity-check note.

    python -m scripts.t307_failures --task 2 [--sample 20]

For every `transfer`/`transfer_mt` condition with both a source (in-language
ceiling, T-302) and target (T-305) prediction log on disk, joins them
(T-306), filters to the mismatch rows, and writes
`data/failures/task_{n}/{condition}.parquet`. Then samples `--sample` rows
across everything exported, pulls the actual source/target text from the
frozen corpus, and writes `reports/task_{n}/t307_audit.md` recording what
was checked.

The audit is a *mechanical* check -- does the filter's own arithmetic (source
correct, target wrong, same item, same seed) hold on real rows, and does the
item_id on both sides resolve to real, non-empty text. It is not a
linguistic check: nobody on this project reads Bengali, Telugu or Malayalam
(CLAUDE.md T-114's second limitation), so no claim is made here about
whether any given row is a genuine transfer failure or a translation
artefact -- that needs different evidence, and is Phase 6's job.

CPU only, no model loading -- reads existing prediction logs off disk.
"""

from __future__ import annotations

import argparse
import random

import pandas as pd

from src.corpus_io import read_split
from src.data import DEFAULT_ENCODER
from src.download_dataset.paths import REPO_ROOT
from src.evaluate import _split_lang, load_matrix
from src.failures import export_failures
from src.predictions import read_prediction_log

REPORTS_DIR = REPO_ROOT / "reports"


def _text_for(task: int, block: str, lang: str, origin: str, item_id: str) -> str | None:
    frame = read_split(task, block, lang, origin)
    row = frame[frame["item_id"] == item_id]
    return None if row.empty else str(row.iloc[0]["text"])


def _clip(text: str | None, n: int = 60) -> str:
    if text is None:
        return ""
    return text[:n] + ("…" if len(text) > n else "")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--sample", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--encoder", default=DEFAULT_ENCODER)
    args = parser.parse_args(argv)

    matrix = load_matrix(args.task)
    # The default encoder keeps the un-namespaced paths its committed artefacts
    # already occupy, and that Phase 6 reads from; others get their own level.
    path_encoder = None if args.encoder == DEFAULT_ENCODER else args.encoder
    exported = []
    all_failures = []
    for condition in matrix["conditions"]:
        if condition["kind"] not in ("transfer", "transfer_mt"):
            continue
        block, lang = _split_lang(condition["train"])
        source_id = f"{block}_{lang}_native_ceiling"
        target_id = condition["name"]
        try:
            source = read_prediction_log(args.task, source_id, encoder=path_encoder)
            target = read_prediction_log(args.task, target_id, encoder=path_encoder)
        except FileNotFoundError as e:
            print(f"skipping {target_id}: {e}")
            continue

        path, written = export_failures(
            args.task, target_id, source, target, encoder=path_encoder
        )
        exported.append({"condition": target_id, "n_failures": len(written), "path": str(path)})
        print(f"{target_id}: {len(written)} failures -> {path}")
        if len(written):
            all_failures.append(written.assign(_source_block=block, _source_lang=lang))

    print()
    print(pd.DataFrame(exported).to_string(index=False))

    if not all_failures:
        print("\nnothing exported; skipping the sanity-check sample")
        return 0

    pool = pd.concat(all_failures, ignore_index=True)
    rng = random.Random(args.seed)
    sample_idx = rng.sample(range(len(pool)), min(args.sample, len(pool)))
    sample = pool.iloc[sample_idx]

    lines = [
        f"# T-307 audit — task {args.task}",
        "",
        f"Sampled {len(sample)} of {len(pool)} exported failure rows across "
        f"{len(exported)} condition(s), seed {args.seed}.",
        "",
        "**Scope: mechanical, not linguistic.** This confirms the filter's "
        "own arithmetic (source correct, target wrong, same item, same seed) "
        "holds on real rows and that both sides resolve to real corpus text. "
        "Nobody on this project reads Bengali, Telugu or Malayalam, so this "
        "note makes no claim about whether any of these failures are "
        "*meaningful* transfer failures versus translation artefacts — see "
        "CLAUDE.md's second T-114 limitation.",
        "",
        "| item_id | seed | gold | pred_src | pred_tgt | src text (Hindi) | tgt text |",
        "|---|---|---|---|---|---|---|",
    ]
    all_ok = True
    for _, row in sample.iterrows():
        src_text = _text_for(args.task, row["_source_block"], row["_source_lang"], "native", row["item_id"])
        tgt_text = _text_for(args.task, row["block_id"], row["lang"], row["origin"], row["item_id"])
        ok = (
            row["pred_src"] == row["gold"]
            and row["pred_tgt"] != row["gold"]
            and src_text is not None
            and tgt_text is not None
        )
        all_ok = all_ok and ok
        lines.append(
            f"| {row['item_id']} | {row['seed']} | {row['gold']} | {row['pred_src']} | "
            f"{row['pred_tgt']} | {_clip(src_text)} | {_clip(tgt_text)} |"
        )

    lines += [
        "",
        f"**Result: {'all' if all_ok else 'NOT all'} {len(sample)} sampled rows pass the "
        "mechanical check** (predicate holds on real values, both item_ids resolve to "
        "non-empty corpus text).",
    ]

    out_dir = REPORTS_DIR / f"task_{args.task}"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if path_encoder is None else f"_{args.encoder}"
    out_path = out_dir / f"t307_audit{suffix}.md"
    out_path.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
