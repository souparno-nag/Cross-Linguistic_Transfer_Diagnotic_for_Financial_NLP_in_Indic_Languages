"""Tests for T-302's per-instance prediction log.

The fast block checks the parts that need no model or corpus: the log's fixed
column shape, the write/read round trip and its per-seed upsert behaviour, and
the 4-way join validator actually catching a missing arm or a stray null. The
`slow` block is the acceptance criterion itself: predictions for all four of a
real block's arms, built from a real checkpoint, join cleanly on `item_id`.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import predictions as P


# --------------------------------------------------------------------------
# Fast -- no model or corpus access
# --------------------------------------------------------------------------


def _fake_result(n: int = 4, k: int = 3):
    return {
        "item_id": [f"it{i}" for i in range(n)],
        "gold": [i % k for i in range(n)],
        "pred": [(i + 1) % k for i in range(n)],
        "probs": [[0.1, 0.2, 0.7] for _ in range(n)],
    }


def _fake_frame(n: int = 4, src_lang=None):
    return pd.DataFrame(
        {
            "item_id": [f"it{i}" for i in range(n)],
            "src_lang": [src_lang] * n,
            "text": [f"text {i}" for i in range(n)],
        }
    )


def test_prediction_rows_has_the_fixed_column_shape_and_order():
    rows = P._prediction_rows(
        _fake_result(),
        _fake_frame(src_lang="hin"),
        condition_id="c1",
        encoder_id="indicbert-v2", run_id="r1",
        block="H",
        lang="ben",
        origin="mt",
        seed=0,
    )
    assert list(rows.columns) == P.PREDICTION_COLUMNS
    assert len(rows) == 4
    assert (rows["condition_id"] == "c1").all()
    assert (rows["src_lang"] == "hin").all()
    assert rows["gold"].tolist() == [0, 1, 2, 0]


def test_prediction_rows_looks_up_src_lang_per_item_not_a_single_value():
    frame = _fake_frame(n=2)
    frame["src_lang"] = ["hin", "ben"]  # deliberately not uniform
    rows = P._prediction_rows(
        _fake_result(n=2), frame, condition_id="c", encoder_id="indicbert-v2", run_id="r",
        block="H", lang="mal", origin="mt", seed=1,
    )
    assert rows["src_lang"].tolist() == ["hin", "ben"]


def test_write_then_read_round_trips(tmp_path):
    rows = P._prediction_rows(
        _fake_result(), _fake_frame(src_lang="hin"), condition_id="c1", encoder_id="indicbert-v2", run_id="r1",
        block="H", lang="ben", origin="mt", seed=0,
    )
    P.write_prediction_log(rows, 3, "c1", root=tmp_path)
    back = P.read_prediction_log(3, "c1", root=tmp_path)
    assert back["item_id"].tolist() == rows["item_id"].tolist()
    # Parquet round-trips a list column as numpy arrays, not python lists.
    assert [list(p) for p in back["probs"]] == rows["probs"].tolist()


def test_read_missing_log_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="no prediction log"):
        P.read_prediction_log(3, "nope", root=tmp_path)


def test_writing_a_second_seed_accumulates_rather_than_overwrites(tmp_path):
    seed0 = P._prediction_rows(
        _fake_result(), _fake_frame(src_lang="hin"), condition_id="c1", encoder_id="indicbert-v2", run_id="r1",
        block="H", lang="ben", origin="mt", seed=0,
    )
    seed1 = P._prediction_rows(
        _fake_result(), _fake_frame(src_lang="hin"), condition_id="c1", encoder_id="indicbert-v2", run_id="r1",
        block="H", lang="ben", origin="mt", seed=1,
    )
    P.write_prediction_log(seed0, 3, "c1", root=tmp_path)
    P.write_prediction_log(seed1, 3, "c1", root=tmp_path)
    combined = P.read_prediction_log(3, "c1", root=tmp_path)
    assert sorted(combined["seed"].unique().tolist()) == [0, 1]
    assert len(combined) == len(seed0) + len(seed1)


def test_rewriting_the_same_seed_replaces_it_instead_of_duplicating(tmp_path):
    original = P._prediction_rows(
        _fake_result(), _fake_frame(src_lang="hin"), condition_id="c1", encoder_id="indicbert-v2", run_id="r1",
        block="H", lang="ben", origin="mt", seed=0,
    )
    P.write_prediction_log(original, 3, "c1", root=tmp_path)

    redone = original.copy()
    redone["pred"] = [0, 0, 0, 0]  # a re-run that scored differently
    P.write_prediction_log(redone, 3, "c1", root=tmp_path)

    combined = P.read_prediction_log(3, "c1", root=tmp_path)
    assert len(combined) == len(original)  # not doubled
    assert combined["pred"].tolist() == [0, 0, 0, 0]  # the new run wins


def test_multiple_arms_of_one_condition_coexist_even_when_item_ids_match(tmp_path):
    """A `translationese` condition writes three arms that share item_id --
    the upsert key must not treat the second arm as a stale duplicate of the
    first just because the items line up."""
    native = P._prediction_rows(
        _fake_result(n=3), _fake_frame(n=3), condition_id="translationese_hin",
        encoder_id="indicbert-v2", run_id="r", block="H", lang="hin", origin="native", seed=0,
    )
    from_ben = P._prediction_rows(
        _fake_result(n=3), _fake_frame(n=3, src_lang="ben"), condition_id="translationese_hin",
        encoder_id="indicbert-v2", run_id="r", block="B", lang="hin", origin="mt", seed=0,
    )
    P.write_prediction_log(native, 2, "translationese_hin", root=tmp_path)
    P.write_prediction_log(from_ben, 2, "translationese_hin", root=tmp_path)
    combined = P.read_prediction_log(2, "translationese_hin", root=tmp_path)
    assert len(combined) == len(native) + len(from_ben)
    assert sorted(combined["block_id"].unique().tolist()) == ["B", "H"]


def test_block_join_succeeds_with_shared_item_ids_and_no_nulls():
    hin = P._prediction_rows(
        _fake_result(n=3), _fake_frame(n=3), condition_id="hin_native_ceiling",
        encoder_id="indicbert-v2", run_id="r", block="H", lang="hin", origin="native", seed=0,
    )
    ben = P._prediction_rows(
        _fake_result(n=3), _fake_frame(n=3, src_lang="hin"), condition_id="hin_to_ben_mt",
        encoder_id="indicbert-v2", run_id="r", block="H", lang="ben", origin="mt", seed=0,
    )
    joined = P.check_block_join({"hin": hin, "ben": ben})
    assert len(joined) == 3
    # src_lang_hin is correctly null: "hin" here is the native arm (§6).
    checked = joined[[c for c in joined.columns if not c.startswith("src_lang_")]]
    assert not checked.isna().any().any()


def test_block_join_raises_when_an_arm_shares_no_items():
    hin = P._prediction_rows(
        _fake_result(n=3), _fake_frame(n=3), condition_id="a", encoder_id="indicbert-v2", run_id="r",
        block="H", lang="hin", origin="native", seed=0,
    )
    other = hin.copy()
    other["item_id"] = [f"different{i}" for i in range(3)]
    with pytest.raises(ValueError, match="zero rows"):
        P.check_block_join({"hin": hin, "ben": other})


def test_block_join_raises_on_a_stray_null():
    hin = P._prediction_rows(
        _fake_result(n=3), _fake_frame(n=3), condition_id="a", encoder_id="indicbert-v2", run_id="r",
        block="H", lang="hin", origin="native", seed=0,
    )
    ben = hin.copy()
    ben.loc[0, "pred"] = None
    with pytest.raises(ValueError, match="nulls"):
        P.check_block_join({"hin": hin, "ben": ben})


# --------------------------------------------------------------------------
# slow -- a real checkpoint, a real block, the acceptance criterion
# --------------------------------------------------------------------------

ENCODER = "mbert-base"


@pytest.mark.slow
def test_a_real_blocks_four_arms_join_with_zero_nulls(tmp_path, monkeypatch):
    """T-302's done criterion, on real data: task 3's block T (Telugu-trained)
    is the cheapest block to predict over (532 rows/arm)."""
    from src import checkpoints
    from src.config import RunConfig, run_training
    from src.ids import targets_for_block

    run = RunConfig(
        encoder=ENCODER, task=3, lang="tel", max_len=64, epochs=1, patience=1,
        dev_fraction=0.2, seed=0,
    )
    work_dir = tmp_path / "checkpoints" / "t302_tel_seed0"
    run_training(run, device="cpu", work_dir=work_dir)
    monkeypatch.setattr(checkpoints, "CHECKPOINT_ROOT", tmp_path / "checkpoints")

    arms = [("tel", "native")] + [(lang, "mt") for lang in targets_for_block("T")]
    logs = {}
    for lang, origin in arms:
        frame = P.prediction_log(
            f"T_tel_to_{lang}", "t302_tel_seed0", 3, "T", lang, origin, run.seed,
            device="cpu",
        )
        assert list(frame.columns) == P.PREDICTION_COLUMNS
        logs[lang] = frame

    joined = P.check_block_join(logs)
    assert len(joined) > 0


# --------------------------------------------------------------------------
# Encoder namespacing (CLAUDE5.md rule 1)
# --------------------------------------------------------------------------


def test_encoder_adds_a_path_level_and_none_keeps_the_original(tmp_path):
    legacy = P.log_path(2, "transfer_hin_to_ben", root=tmp_path)
    namespaced = P.log_path(2, "transfer_hin_to_ben", root=tmp_path, encoder="mbert-base")
    assert legacy == tmp_path / "task_2" / "transfer_hin_to_ben.parquet"
    assert namespaced == tmp_path / "task_2" / "mbert-base" / "transfer_hin_to_ben.parquet"
    assert legacy != namespaced


def test_two_encoders_do_not_share_a_file(tmp_path):
    """The collision this exists to prevent: without a level, the second
    encoder's rows merge into the first's file, and results_tables joins on
    (item_id, seed) with no encoder filter."""
    rows = P._prediction_rows(
        _fake_result(n=3), _fake_frame(n=3), condition_id="c",
        encoder_id="indicbert-v2", run_id="r_ib",
        block="H", lang="hin", origin="native", seed=0,
    )
    other = P._prediction_rows(
        _fake_result(n=3), _fake_frame(n=3), condition_id="c",
        encoder_id="mbert-base", run_id="r_mb",
        block="H", lang="hin", origin="native", seed=0,
    )
    P.write_prediction_log(rows, 2, "c", root=tmp_path)
    P.write_prediction_log(other, 2, "c", root=tmp_path, encoder="mbert-base")

    first = P.read_prediction_log(2, "c", root=tmp_path)
    second = P.read_prediction_log(2, "c", root=tmp_path, encoder="mbert-base")
    assert len(first) == 3 and len(second) == 3
    assert set(first["encoder_id"]) == {"indicbert-v2"}
    assert set(second["encoder_id"]) == {"mbert-base"}


def test_encoder_id_is_carried_on_every_row():
    rows = P._prediction_rows(
        _fake_result(n=4), _fake_frame(n=4), condition_id="c",
        encoder_id="mbert-base", run_id="r",
        block="H", lang="hin", origin="native", seed=1,
    )
    assert list(rows.columns) == P.PREDICTION_COLUMNS
    assert (rows["encoder_id"] == "mbert-base").all()
