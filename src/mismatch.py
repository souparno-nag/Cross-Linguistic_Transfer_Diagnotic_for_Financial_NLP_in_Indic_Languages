"""The mismatch filter predicate (CLAUDE3.md T-306).

This is what makes Phase 6 meaningful. A model getting an item wrong in a
target language could mean the language beat it, or it could mean the item
was just hard and the model would have gotten it wrong in the source language
too -- item difficulty and model incapacity are indistinguishable from a
target-language score alone. Restricting to items the model *did* get right,
one language earlier, rules that out by construction: whatever changed
between the source and target arm is the only thing left that could explain
the new failure.

The predicate, row-wise over an item the model scored in both a source and a
target arm:

    ŷ_src == y   AND   ŷ_tgt != y

`join_source_target` does the joining (on `item_id` *and* `seed`, so a
seed-0 source prediction is never paired with a seed-1 target one -- that
would compare two different trained instances of the model, not one model's
behaviour across languages) and asserts the aligned item carries one gold
label on both sides, per §6 / T-102b. `mismatch_mask` is the predicate itself;
`filter_mismatches` and `mismatch_report` are what T-307's export and this
task's own per-condition counts are built from.
"""

from __future__ import annotations

import pandas as pd

JOIN_KEYS = ["item_id", "seed"]


def join_source_target(source: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    """One row per (item, seed) present in both logs, source and target kept
    distinct (`pred_src`/`pred_tgt`), target's descriptive columns
    (`block_id`, `lang`, `origin`, `src_lang`, `probs`) carried through
    unprefixed since they describe *this* row's arm.

    An inner join: an item scored on only one side cannot be checked for this
    pattern at all, so it is dropped rather than padded with a null.
    """
    left = source[[*JOIN_KEYS, "gold", "pred", "condition_id", "run_id"]].rename(
        columns={
            "gold": "gold_src",
            "pred": "pred_src",
            "condition_id": "src_condition_id",
            "run_id": "src_run_id",
        }
    )
    right = target.rename(
        columns={
            "gold": "gold_tgt",
            "pred": "pred_tgt",
            "condition_id": "tgt_condition_id",
            "run_id": "tgt_run_id",
        }
    )
    merged = left.merge(right, on=JOIN_KEYS, how="inner")

    disagree = merged["gold_src"] != merged["gold_tgt"]
    if disagree.any():
        bad = merged.loc[disagree, "item_id"].tolist()[:5]
        raise ValueError(
            f"{int(disagree.sum())} (item, seed) pairs carry a different gold "
            f"label in the source and target log, e.g. {bad} -- an aligned item "
            "must carry one label across languages (T-102b); this points at a "
            "data problem upstream, not something to paper over here"
        )
    return merged.drop(columns=["gold_tgt"]).rename(columns={"gold_src": "gold"})


def mismatch_mask(joined: pd.DataFrame) -> pd.Series:
    """ŷ_src == y AND ŷ_tgt != y — correct one language earlier, wrong here."""
    return (joined["pred_src"] == joined["gold"]) & (joined["pred_tgt"] != joined["gold"])


def filter_mismatches(joined: pd.DataFrame) -> pd.DataFrame:
    """Just the rows the predicate keeps -- T-307's export starts here."""
    return joined[mismatch_mask(joined)].reset_index(drop=True)


def mismatch_report(source: pd.DataFrame, target: pd.DataFrame) -> dict:
    """Per-condition and per-seed mismatch counts, for T-306's own done
    criterion ("per-condition counts reported")."""
    joined = join_source_target(source, target)
    mask = mismatch_mask(joined)
    condition_id = target["condition_id"].iloc[0] if len(target) else None

    by_seed = {}
    if len(joined):
        grouped = joined.assign(mismatch=mask).groupby("seed")["mismatch"]
        for seed, group in grouped:
            by_seed[int(seed)] = {"n_mismatch": int(group.sum()), "n": int(group.count())}

    return {
        "condition_id": condition_id,
        "n_joined": len(joined),
        "n_mismatch": int(mask.sum()),
        "rate": float(mask.mean()) if len(joined) else 0.0,
        "by_seed": by_seed,
    }
