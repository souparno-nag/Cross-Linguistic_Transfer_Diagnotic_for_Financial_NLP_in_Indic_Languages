"""Tests for T-307's failed-instance export.

Fast, hand-built fixtures -- no model or corpus access.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import failures as F


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


def test_failure_path_carries_a_task_level():
    a = F.failure_path(2, "transfer_hin_to_ben")
    b = F.failure_path(3, "transfer_hin_to_ben")
    assert a != b
    assert "task_2" in a.parts and "task_3" in b.parts


def test_export_writes_only_the_mismatch_rows(tmp_path):
    source = _log(
        ["a", "b", "c", "d"], gold=[0, 1, 0, 1], pred=[0, 1, 1, 0],
        condition_id="H_hin_native_ceiling", run_id="r_seed0", block="H", lang="hin", origin="native",
    )
    target = _log(
        ["a", "b", "c", "d"], gold=[0, 1, 0, 1], pred=[1, 1, 1, 1],
        condition_id="transfer_hin_to_ben", run_id="r_seed0",
    )
    path, written = F.export_failures(2, "transfer_hin_to_ben", source, target, root=tmp_path)
    assert path.exists()
    assert written["item_id"].tolist() == ["a"]

    back = F.read_failures(2, "transfer_hin_to_ben", root=tmp_path)
    assert back["item_id"].tolist() == ["a"]
    assert back.iloc[0]["pred_src"] == back.iloc[0]["gold"] == 0
    assert back.iloc[0]["pred_tgt"] != back.iloc[0]["gold"]


def test_export_with_zero_mismatches_still_writes_an_empty_file(tmp_path):
    """Writing nothing at all would look identical to "not yet run" -- an
    empty file on disk is the honest way to record "checked, found none"."""
    source = _log(["a"], gold=[0], pred=[0], condition_id="src", run_id="r", block="H", lang="hin", origin="native")
    target = _log(["a"], gold=[0], pred=[0], condition_id="tgt", run_id="r")  # correct in both
    path, written = F.export_failures(2, "always_correct", source, target, root=tmp_path)
    assert path.exists()
    assert len(written) == 0
    back = F.read_failures(2, "always_correct", root=tmp_path)
    assert len(back) == 0
    assert list(back.columns) == list(written.columns)


def test_read_missing_failure_set_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no failure set"):
        F.read_failures(2, "nope", root=tmp_path)


def test_export_raises_on_conflicting_gold_the_same_way_the_join_does(tmp_path):
    """export_failures doesn't re-implement the T-306 safety check -- it
    reuses join_source_target, so a labelling conflict still surfaces here."""
    source = _log(["a"], gold=[0], pred=[0], condition_id="src", run_id="r", block="H", lang="hin", origin="native")
    target = _log(["a"], gold=[1], pred=[1], condition_id="tgt", run_id="r")
    with pytest.raises(ValueError, match="different gold label"):
        F.export_failures(2, "broken", source, target, root=tmp_path)
