"""Tests for T-306's mismatch filter predicate: ŷ_src == y AND ŷ_tgt != y.

All hand-built fixtures, no model or corpus access, per T-306's own done
criterion ("unit-tested against hand-built fixtures; per-condition counts
reported").
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import mismatch as M


def _log(item_ids, gold, pred, *, condition_id, run_id, seed=0, block="H", lang="ben", origin="mt"):
    n = len(item_ids)
    return pd.DataFrame(
        {
            "condition_id": [condition_id] * n,
            "run_id": [run_id] * n,
            "item_id": item_ids,
            "block_id": [block] * n,
            "lang": [lang] * n,
            "origin": [origin] * n,
            "src_lang": ["hin"] * n if origin == "mt" else [None] * n,
            "gold": gold,
            "pred": pred,
            "probs": [[0.5, 0.5]] * n,
            "seed": [seed] * n,
        }
    )


def test_the_hand_worked_example_from_the_spec():
    # item a: right in source, wrong in target -> mismatch.
    # item b: right in both -> not a mismatch.
    # item c: wrong in source, wrong in target -> not a mismatch (incapacity,
    #         not a transfer failure -- the model never had it).
    # item d: wrong in source, right in target -> not a mismatch.
    source = _log(
        ["a", "b", "c", "d"], gold=[0, 1, 0, 1], pred=[0, 1, 1, 0],
        condition_id="H_hin_native_ceiling", run_id="r_seed0", block="H", lang="hin", origin="native",
    )
    target = _log(
        ["a", "b", "c", "d"], gold=[0, 1, 0, 1], pred=[1, 1, 1, 1],
        condition_id="transfer_hin_to_ben", run_id="r_seed0",
    )
    joined = M.join_source_target(source, target)
    kept = M.filter_mismatches(joined)
    assert kept["item_id"].tolist() == ["a"]


def test_mismatch_mask_matches_a_manually_computed_boolean_array():
    joined = pd.DataFrame(
        {
            "item_id": ["a", "b", "c", "d", "e"],
            "seed": [0, 0, 0, 0, 0],
            "gold": [0, 1, 2, 0, 1],
            "pred_src": [0, 1, 2, 1, 0],  # right, right, right, wrong, wrong
            "pred_tgt": [1, 1, 2, 1, 1],  # wrong, right, right, wrong, wrong
        }
    )
    mask = M.mismatch_mask(joined)
    # only item "a": src correct (0==0), tgt wrong (1!=0)
    assert mask.tolist() == [True, False, False, False, False]


def test_join_keeps_seeds_paired_rather_than_crossing_them():
    """A seed-0 source prediction must never pair with a seed-1 target one --
    that would compare two different trained instances of the model."""
    source = pd.concat(
        [
            _log(["a"], gold=[0], pred=[0], seed=0, condition_id="src", run_id="r_seed0", block="H", lang="hin", origin="native"),
            _log(["a"], gold=[0], pred=[1], seed=1, condition_id="src", run_id="r_seed1", block="H", lang="hin", origin="native"),
        ],
        ignore_index=True,
    )
    target = _log(["a"], gold=[0], pred=[1], seed=0, condition_id="tgt", run_id="r_seed0")
    joined = M.join_source_target(source, target)
    # only the seed-0 pair should have matched; the seed-1 source row has
    # no seed-1 target row to join against, so it is dropped by the inner join.
    assert len(joined) == 1
    assert joined.iloc[0]["seed"] == 0
    assert joined.iloc[0]["pred_src"] == 0  # the seed-0 source prediction, not seed-1's


def test_conflicting_gold_between_source_and_target_raises():
    source = _log(["a"], gold=[0], pred=[0], condition_id="src", run_id="r", block="H", lang="hin", origin="native")
    target = _log(["a"], gold=[1], pred=[1], condition_id="tgt", run_id="r")  # same item, different label
    with pytest.raises(ValueError, match="different gold label"):
        M.join_source_target(source, target)


def test_an_item_missing_from_one_side_is_dropped_not_padded():
    source = _log(["a", "b"], gold=[0, 1], pred=[0, 1], condition_id="src", run_id="r", block="H", lang="hin", origin="native")
    target = _log(["a"], gold=[0], pred=[1], condition_id="tgt", run_id="r")  # "b" absent
    joined = M.join_source_target(source, target)
    assert joined["item_id"].tolist() == ["a"]


# --------------------------------------------------------------------------
# mismatch_report: per-condition, per-seed counts
# --------------------------------------------------------------------------


def test_mismatch_report_counts_and_rate():
    source = _log(
        ["a", "b", "c", "d"], gold=[0, 1, 0, 1], pred=[0, 1, 1, 0],
        condition_id="H_hin_native_ceiling", run_id="r_seed0", block="H", lang="hin", origin="native",
    )
    target = _log(
        ["a", "b", "c", "d"], gold=[0, 1, 0, 1], pred=[1, 1, 1, 1],
        condition_id="transfer_hin_to_ben", run_id="r_seed0",
    )
    report = M.mismatch_report(source, target)
    assert report["condition_id"] == "transfer_hin_to_ben"
    assert report["n_joined"] == 4
    assert report["n_mismatch"] == 1
    assert report["rate"] == pytest.approx(0.25)


def test_mismatch_report_breaks_down_by_seed():
    source = pd.concat(
        [
            _log(["a", "b"], gold=[0, 1], pred=[0, 1], seed=0, condition_id="src", run_id="r0", block="H", lang="hin", origin="native"),
            _log(["a", "b"], gold=[0, 1], pred=[0, 0], seed=1, condition_id="src", run_id="r1", block="H", lang="hin", origin="native"),
        ],
        ignore_index=True,
    )
    target = pd.concat(
        [
            _log(["a", "b"], gold=[0, 1], pred=[1, 1], seed=0, condition_id="tgt", run_id="r0"),
            _log(["a", "b"], gold=[0, 1], pred=[1, 1], seed=1, condition_id="tgt", run_id="r1"),
        ],
        ignore_index=True,
    )
    report = M.mismatch_report(source, target)
    # seed 0: src correct on a (0==0) and b (1==1); tgt wrong on a (1!=0) but
    #         right on b (1==1) -> only "a" is a mismatch.
    # seed 1: src correct on a (0==0) only, wrong on b (0!=1); tgt wrong on
    #         both -> only "a" (src right, tgt wrong) is a mismatch.
    assert report["by_seed"][0] == {"n_mismatch": 1, "n": 2}
    assert report["by_seed"][1] == {"n_mismatch": 1, "n": 2}
    assert report["n_mismatch"] == 2


def test_mismatch_report_on_empty_join_is_zero_not_an_error():
    source = _log(["a"], gold=[0], pred=[0], condition_id="src", run_id="r", block="H", lang="hin", origin="native")
    target = _log(["z"], gold=[0], pred=[0], condition_id="tgt", run_id="r")
    report = M.mismatch_report(source, target)
    assert report == {
        "condition_id": "tgt",
        "n_joined": 0,
        "n_mismatch": 0,
        "rate": 0.0,
        "by_seed": {},
    }
