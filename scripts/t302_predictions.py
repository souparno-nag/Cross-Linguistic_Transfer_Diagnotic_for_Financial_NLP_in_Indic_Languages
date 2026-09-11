"""T-302 — build the per-instance prediction log for one checkpoint's own block.

    python -m scripts.t302_predictions --task 2 --run-id task2_hin_indicbert_seed0

For a checkpoint trained on a block's native split, predicts over all four of
that block's language arms (the native split plus its three MT translations —
§6, every block is 4-way parallel) and writes one `data/predictions/{condition
_id}.parquet` per arm. The native arm is the model's own training text, kept
only as the explicitly-labelled in-language ceiling hard rule 4 allows, never
as a transfer result.

Condition ids are taken from `configs/eval_conditions.json` where a matching
`transfer_mt` condition exists (this run's split as `train`, the arm's split
as `eval`); the native arm has no such condition; done: enforced afterwards
that the four arms join on `item_id` with zero nulls (T-302's acceptance).
"""

from __future__ import annotations

import argparse
import json
import sys

from src.conditions import DEFAULT_SEED, build as build_conditions
from src.download_dataset.paths import REPO_ROOT
from src.ids import targets_for_block
from src.inference import load_run_config_for
from src.predictions import check_block_join, prediction_log, write_prediction_log


def _condition_id_for(task: int, train_split: str, eval_split: str) -> str | None:
    matrix = build_conditions(task, seed=DEFAULT_SEED)
    for condition in matrix["conditions"]:
        if condition.get("train") == train_split and condition.get("eval") == eval_split:
            return condition["name"]
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    run = load_run_config_for(args.run_id)
    block = run.resolved_block()
    native = run.lang
    # RunConfig.split_id() reads "taskN/BLOCK/lang/origin"; eval_conditions.json
    # keys its splits as "task_N/BLOCK/lang" -- build that spelling directly.
    train_split = f"task_{args.task}/{block}/{native}"

    arms = [(native, "native")] + [(lang, "mt") for lang in targets_for_block(block)]
    logs: dict[str, "object"] = {}

    for lang, origin in arms:
        eval_split = f"task_{args.task}/{block}/{lang}"
        if origin == "native":
            condition_id = f"{block}_{lang}_native_ceiling"
            print(
                f"{condition_id}: in-language ceiling (hard rule 4) -- the "
                "model's own training text, not a transfer result"
            )
        else:
            condition_id = _condition_id_for(args.task, train_split, eval_split)
            if condition_id is None:
                print(f"WARNING: no matching condition for {train_split} -> {eval_split}", file=sys.stderr)
                condition_id = f"{block}_{native}_to_{lang}_mt_unregistered"

        frame = prediction_log(
            condition_id, args.run_id, args.task, block, lang, origin, run.seed,
            device=args.device, batch_size=args.batch_size,
        )
        path = write_prediction_log(frame, args.task, condition_id)
        print(f"wrote {len(frame)} rows -> {path.relative_to(REPO_ROOT)}")
        logs[lang] = frame

    joined = check_block_join(logs)
    print(f"\nblock {block} joins 4-way on item_id: {len(joined)} items, zero nulls.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
