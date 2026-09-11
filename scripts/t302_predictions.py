"""T-302 — build the in-language-ceiling prediction log, and prove the block
still joins 4-way.

    python -m scripts.t302_predictions --task 2 --run-id task2_hin_indicbert_seed0

For a checkpoint trained on a block's native split, predicts over all four of
that block's language arms (the native split plus its three MT translations —
§6, every block is 4-way parallel) and checks they join on `item_id` with
zero nulls (T-302's acceptance criterion). Only the **native** arm is
persisted, to `data/predictions/task_{n}/{block}_{lang}_native_ceiling.
parquet` -- it is the model's own training text, kept only as the
explicitly-labelled in-language ceiling hard rule 4 allows, and it is what
T-306/T-308 pair against a target arm to compute a transfer gap.

The three MT arms are predicted only in memory, to prove the join, and are
**not** written to disk here. `transfer_hin_to_ben_mt` and its siblings are
T-305's condition names (`configs/eval_conditions.json`'s `transfer_mt`
family) and T-305 restricts them to the `eval_items: "partition:eval"` half
of the split before predicting (evaluate.py's `item_filter`). Persisting an
*unfiltered* version of the same file here — as an earlier version of this
script did — silently reintroduced exactly the item-partition leak T-305
was fixed for: run this CLI for a second or third seed and it overwrote
T-305's correctly-filtered rows with unfiltered ones for those seeds only,
which is how it was actually caught (`transfer_hin_to_ben_mt.parquet` ended
up with 1117 correct seed-0 rows next to 2238 unfiltered seed-1/2 rows).
"""

from __future__ import annotations

from src.download_dataset.paths import REPO_ROOT
from src.ids import targets_for_block
from src.inference import load_run_config_for
from src.predictions import check_block_join, prediction_log, write_prediction_log


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args(argv)

    run = load_run_config_for(args.run_id)
    block = run.resolved_block()
    native = run.lang

    arms = [(native, "native")] + [(lang, "mt") for lang in targets_for_block(block)]
    logs: dict[str, "object"] = {}

    for lang, origin in arms:
        if origin == "native":
            condition_id = f"{block}_{lang}_native_ceiling"
            print(
                f"{condition_id}: in-language ceiling (hard rule 4) -- the "
                "model's own training text, not a transfer result"
            )
        else:
            condition_id = f"{block}_{native}_to_{lang}_mt_join_check"

        frame = prediction_log(
            condition_id, args.run_id, args.task, block, lang, origin, run.seed,
            device=args.device, batch_size=args.batch_size,
        )
        if origin == "native":
            path = write_prediction_log(frame, args.task, condition_id)
            print(f"wrote {len(frame)} rows -> {path.relative_to(REPO_ROOT)}")
        else:
            print(f"predicted {len(frame)} rows for {lang} (join check only, not written)")
        logs[lang] = frame

    joined = check_block_join(logs)
    print(f"\nblock {block} joins 4-way on item_id: {len(joined)} items, zero nulls.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
