"""Per-instance prediction log (CLAUDE3.md T-302).

This is the artefact Phase 6 actually consumes — it needs every prediction,
not summary metrics, so it can filter for the specific mismatch pattern
(correct in the source language, wrong in the target) that T-306 defines.
Losing a row here is not recoverable by re-running an aggregate.

Row schema, fixed by CLAUDE3.md's Outputs table:

    condition_id, run_id, item_id, block_id, lang, origin, src_lang,
    gold, pred, probs, seed

`_prediction_rows` is the pure assembly step (no model or corpus access), so
it is cheap to test on its own; `prediction_log` is the thin wrapper that
loads the checkpoint and the split and calls it.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import SplitDataset, get_tokenizer, load_split
from .download_dataset.paths import REPO_ROOT
from .ids import join_languages
from .inference import LoadedRun, load_frozen_model, predict

PREDICTIONS_ROOT = REPO_ROOT / "data" / "predictions"

PREDICTION_COLUMNS = [
    "condition_id",
    "run_id",
    "item_id",
    "block_id",
    "lang",
    "origin",
    "src_lang",
    "gold",
    "pred",
    "probs",
    "seed",
]


def _prediction_rows(
    result: dict,
    frame: pd.DataFrame,
    *,
    condition_id: str,
    run_id: str,
    block: str,
    lang: str,
    origin: str,
    seed: int,
) -> pd.DataFrame:
    """Assemble the fixed-column log from `predict()`'s raw output.

    `src_lang` is looked up from the split itself rather than threaded through
    `predict`, since it is a per-row corpus fact (§6), not a model output.
    """
    src_lang_by_item = dict(zip(frame["item_id"], frame["src_lang"]))
    n = len(result["item_id"])
    rows = pd.DataFrame(
        {
            "condition_id": [condition_id] * n,
            "run_id": [run_id] * n,
            "item_id": result["item_id"],
            "block_id": [block] * n,
            "lang": [lang] * n,
            "origin": [origin] * n,
            "src_lang": [src_lang_by_item[i] for i in result["item_id"]],
            "gold": result["gold"],
            "pred": result["pred"],
            "probs": result["probs"],
            "seed": [seed] * n,
        }
    )
    return rows[PREDICTION_COLUMNS]


def prediction_log_from_loaded(
    loaded: LoadedRun,
    condition_id: str,
    task: int,
    block: str,
    lang: str,
    origin: str,
    seed: int,
    *,
    batch_size: int = 32,
    item_filter=None,
) -> pd.DataFrame:
    """Like `prediction_log`, but reuses an already-loaded checkpoint.

    T-305 evaluates several conditions (and, for `translationese`, several
    arms of one condition) from the same checkpoint; reloading the encoder
    for each one would be correct but wasteful. Callers that only need one
    prediction still go through `prediction_log`.

    `item_filter`, if given, is a ``str -> bool`` predicate applied to
    `item_id` before tokenising -- this is how T-305 restricts a `transfer`/
    `transfer_mt` condition to its `eval_items: "partition:eval"` half rather
    than predicting over the whole split, some of which the model's own
    training language literally trained on (tasks 2/3 share `item_id` across
    languages for an aligned item, T-102b).
    """
    frame = load_split(task, block, lang, origin)
    if item_filter is not None:
        frame = frame[frame["item_id"].map(item_filter)].reset_index(drop=True)
        if frame.empty:
            raise ValueError(
                f"item_filter left zero rows for task_{task}/{block}/{lang}/{origin}"
            )
    tokenizer = get_tokenizer(loaded.run_config.encoder)
    dataset = SplitDataset(frame, tokenizer, max_len=loaded.run_config.max_len)
    result = predict(loaded, dataset, batch_size=batch_size)
    return _prediction_rows(
        result,
        frame,
        condition_id=condition_id,
        run_id=loaded.run_id,
        block=block,
        lang=lang,
        origin=origin,
        seed=seed,
    )


def prediction_log(
    condition_id: str,
    run_id: str,
    task: int,
    block: str,
    lang: str,
    origin: str,
    seed: int,
    *,
    device: str = "cpu",
    batch_size: int = 32,
) -> pd.DataFrame:
    """One condition's per-instance predictions from a frozen checkpoint.

    `task`/`block`/`lang`/`origin` name the split being evaluated; they need
    not match the checkpoint's own training split (that is the point of
    zero-shot evaluation). `condition_id` and `seed` are caller-supplied
    bookkeeping — this module does not look them up in `eval_conditions.json`,
    since it has no opinion on which conditions exist, only on the fixed shape
    every condition's log must have.
    """
    loaded = load_frozen_model(run_id, device=device)
    return prediction_log_from_loaded(
        loaded, condition_id, task, block, lang, origin, seed, batch_size=batch_size
    )


# --------------------------------------------------------------------------
# Storage: one Parquet file per condition, seeds accumulate
# --------------------------------------------------------------------------


def log_path(task: int, condition_id: str, *, root: Path | None = None) -> Path:
    """`data/predictions/task_{n}/{condition_id}.parquet`.

    The `task_{n}` level is not decoration (CLAUDE.md §5): task 2 and task 3
    both produce a condition named e.g. `transfer_hin_to_ben_mt`, and without
    this level the two tasks' logs collide on one path and silently merge --
    exactly the corruption §5 warns about for corpus data, and just as
    unrecoverable here since item_id is only unique *within* a task.
    """
    root = root if root is not None else PREDICTIONS_ROOT
    return root / f"task_{task}" / f"{condition_id}.parquet"


def write_prediction_log(
    frame: pd.DataFrame, task: int, condition_id: str, *, root: Path | None = None
) -> Path:
    """Write one condition's log, merging with whatever is already on disk.

    T-305 runs 3 seeds per condition (hard rule 5), so calls for the same
    condition accumulate rather than overwrite. Keyed on `(item_id, seed,
    run_id, block_id, lang, origin)`: re-running one seed/arm replaces just
    those rows instead of duplicating them or losing the others. The arm
    columns matter because a `translationese` condition writes three arms
    (native plus two MT provenances) that all share `item_id` -- keying on
    `item_id` alone would make the second and third arm's rows look like
    stale duplicates of the first and silently drop them.
    """
    path = log_path(task, condition_id, root=root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = pd.read_parquet(path)
        key = ["item_id", "seed", "run_id", "block_id", "lang", "origin"]
        stale = existing.set_index(key).index.isin(frame.set_index(key).index)
        frame = pd.concat([existing.loc[~stale], frame], ignore_index=True)
    frame.to_parquet(path, index=False)
    return path


def read_prediction_log(
    task: int, condition_id: str, *, root: Path | None = None
) -> pd.DataFrame:
    path = log_path(task, condition_id, root=root)
    if not path.exists():
        raise FileNotFoundError(f"no prediction log for condition {condition_id!r} at {path}")
    return pd.read_parquet(path)


# --------------------------------------------------------------------------
# T-302's done criterion: one block's arms join 4-way, zero nulls
# --------------------------------------------------------------------------


def check_block_join(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """The join Phase 6 needs: did this model call an item right in one
    language and wrong in another. Every arm of a block shares `item_id` (§6),
    so that join only works if it is complete and clean.

    `frames` maps an arbitrary key (a split name, an arm label) to that arm's
    prediction log. Raises if the join is empty or carries any null -- except
    in `src_lang`, which is *correctly* null for a native arm (§6) and would
    otherwise fail this check on every block, native row or not.
    """
    joined = join_languages(frames, on="item_id")
    if joined.empty:
        raise ValueError(
            "block join produced zero rows -- no item_id is shared across every "
            "arm passed in"
        )
    checked = joined[[c for c in joined.columns if not c.startswith("src_lang_")]]
    null_cols = checked.columns[checked.isna().any()].tolist()
    if null_cols:
        raise ValueError(f"block join has nulls in {null_cols}")
    return joined
