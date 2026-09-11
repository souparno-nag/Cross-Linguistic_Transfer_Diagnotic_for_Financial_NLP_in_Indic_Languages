"""Tests for T-305's evaluation-condition runner.

The fast block checks the pure planning logic against the real, committed
`configs/eval_conditions.json` -- no model or corpus access, just deciding
which (condition, seed) pairs are runnable given whichever checkpoints exist.
The `slow` block is the acceptance criterion itself: a runnable condition,
run end to end, leaves a row in `experiments.csv` and a prediction log on disk.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import evaluate as E


# --------------------------------------------------------------------------
# Fast -- no model or corpus access
# --------------------------------------------------------------------------


def test_run_id_naming_matches_the_shipped_checkpoint_convention():
    assert E.run_id_for(2, "hin", 0) == "task2_hin_indicbert_seed0"
    assert E.run_id_for(3, "tel", 2) == "task3_tel_indicbert_seed2"


def test_split_lang_parses_the_matrix_spelling():
    assert E._split_lang("task_2/H/hin") == ("H", "hin")
    assert E._split_lang("task_3/T/mal") == ("T", "mal")


@pytest.mark.parametrize(
    "block,lang,expected",
    [("H", "hin", "native"), ("H", "ben", "mt"), ("T", "tel", "native"), ("B", "mal", "mt")],
)
def test_origin_for_matches_the_blocks_own_native_language(block, lang, expected):
    assert E._origin_for(block, lang) == expected


def test_model_lang_for_a_transfer_condition_is_the_training_language():
    condition = {"kind": "transfer", "train": "task_2/B/ben", "eval": "task_2/H/hin"}
    assert E._model_lang(condition) == "ben"


def test_model_lang_for_a_transfer_mt_condition_is_the_training_language():
    condition = {"kind": "transfer_mt", "train": "task_2/T/tel", "eval": "task_2/T/mal"}
    assert E._model_lang(condition) == "tel"


def test_model_lang_for_translationese_is_the_languages_own_native_arm():
    condition = {
        "kind": "translationese",
        "train": None,
        "eval": ["task_2/H/hin", "task_2/B/hin", "task_2/T/hin"],
    }
    assert E._model_lang(condition) == "hin"


def test_plan_covers_every_condition_times_every_seed():
    matrix = E.load_matrix(3)
    entries = E.plan(3, matrix, seeds=(0, 1, 2))
    assert len(entries) == len(matrix["conditions"]) * 3
    names = {(e["condition"], e["seed"]) for e in entries}
    assert names == {(c["name"], s) for c in matrix["conditions"] for s in (0, 1, 2)}


def test_plan_marks_runnable_exactly_where_a_checkpoint_exists(monkeypatch):
    from src import checkpoints

    monkeypatch.setattr(checkpoints, "exists", lambda run_id: run_id.startswith("task3_hin_"))
    entries = E.plan(3)
    runnable_langs = {e["model_lang"] for e in entries if e["runnable"]}
    blocked_langs = {e["model_lang"] for e in entries if not e["runnable"]}
    assert runnable_langs == {"hin"}
    assert blocked_langs == {"ben", "tel"}


def test_plan_with_no_checkpoints_at_all_reports_everything_blocked(monkeypatch):
    from src import checkpoints

    monkeypatch.setattr(checkpoints, "exists", lambda run_id: False)
    entries = E.plan(2)
    assert all(not e["runnable"] for e in entries)


def test_load_matrix_raises_for_a_task_with_no_entry(tmp_path, monkeypatch):
    empty = tmp_path / "eval_conditions.json"
    empty.write_text("{}")
    monkeypatch.setattr(E, "CONDITIONS_PATH", empty)
    with pytest.raises(KeyError, match="task_9"):
        E.load_matrix(9)


# --------------------------------------------------------------------------
# slow -- a real checkpoint, one condition, the acceptance criterion
# --------------------------------------------------------------------------

ENCODER = "mbert-base"


@pytest.mark.slow
def test_a_runnable_condition_logs_a_row_and_a_prediction_file(tmp_path, monkeypatch):
    """T-305's done criterion: a runnable condition gets a result row in
    experiments.csv, plus its per-instance predictions on disk (T-302)."""
    from src import checkpoints, predictions
    from src.config import RunConfig, run_training

    run = RunConfig(
        encoder=ENCODER, task=3, lang="tel", max_len=64, epochs=1, patience=1,
        dev_fraction=0.2, seed=0,
    )
    checkpoints_root = tmp_path / "checkpoints"
    run_training(run, device="cpu", work_dir=checkpoints_root / "task3_tel_indicbert_seed0")
    monkeypatch.setattr(checkpoints, "CHECKPOINT_ROOT", checkpoints_root)
    # Redirect prediction-log output so this test never touches the real
    # data/predictions/ tree.
    monkeypatch.setattr(predictions, "PREDICTIONS_ROOT", tmp_path / "predictions")

    condition = {
        "name": "test_transfer_tel_to_ben",
        "kind": "transfer",
        "train": "task_3/T/tel",
        "eval": "task_3/B/ben",
    }
    results_path = tmp_path / "experiments.csv"
    rows = E.run_condition(
        3, condition, "task3_tel_indicbert_seed0", 0,
        device="cpu", n_boot=50, results_path=results_path,
    )

    assert len(rows) == 1
    assert rows[0]["split"] == "test_transfer_tel_to_ben::task_3/B/ben"
    assert 0.0 <= rows[0]["macro_f1"] <= 1.0
    assert rows[0]["ci_low"] <= rows[0]["macro_f1"] <= rows[0]["ci_high"]

    logged = pd.read_csv(results_path)
    assert len(logged) == 1
    assert logged.iloc[0]["run_id"] == "task3_tel_indicbert_seed0__test_transfer_tel_to_ben__ben"

    log = predictions.read_prediction_log(3, "test_transfer_tel_to_ben")
    assert len(log) > 0
    assert (log["block_id"] == "B").all() and (log["lang"] == "ben").all()
