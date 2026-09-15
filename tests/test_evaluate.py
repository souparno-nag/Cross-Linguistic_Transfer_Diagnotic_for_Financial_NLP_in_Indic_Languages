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


def test_item_filter_is_none_for_a_condition_evaluated_on_the_whole_split():
    """`translationese`'s `eval_items: "all"` is deliberate (its native arm
    is the training text, labelled explicitly per hard rule 4) and must not
    be filtered."""
    condition = {"kind": "translationese", "eval_items": "all"}
    assert E._item_filter_for(condition, 12345) is None


def test_item_filter_restricts_to_the_eval_half_of_the_partition():
    """Regression: run_condition used to predict over a transfer/transfer_mt
    condition's *entire* split, including items whose item_id was in the
    model's own training half (tasks 2/3 share item_id across languages for
    an aligned item, T-102b) -- silently contaminating every "zero-shot"
    number with items the model had effectively already seen."""
    from src.conditions import partition_of

    condition = {"kind": "transfer", "eval_items": "partition:eval"}
    keep = E._item_filter_for(condition, 999)
    assert keep is not None
    sample_ids = [f"t2_{i:06d}" for i in range(200)]
    for item_id in sample_ids:
        assert keep(item_id) == (partition_of(item_id, seed=999) == "eval")
    # and it actually excludes some items, not a no-op filter
    assert any(not keep(i) for i in sample_ids)


def test_item_filter_is_none_for_an_unpartitioned_task():
    condition = {"kind": "transfer", "eval_items": "all"}
    assert E._item_filter_for(condition, 1) is None


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


@pytest.mark.slow
def test_a_partitioned_condition_only_predicts_the_eval_half(tmp_path, monkeypatch):
    """Regression for the item-partition leak: a `transfer` condition with
    `eval_items: "partition:eval"` must predict strictly fewer rows than the
    full target split, and every item_id it does predict must actually fall
    in the eval half."""
    from src import checkpoints, predictions
    from src.config import RunConfig, run_training
    from src.conditions import partition_of
    from src.data import load_native

    run = RunConfig(
        encoder=ENCODER, task=3, lang="tel", max_len=64, epochs=1, patience=1,
        dev_fraction=0.2, seed=0,
    )
    checkpoints_root = tmp_path / "checkpoints"
    run_training(run, device="cpu", work_dir=checkpoints_root / "task3_tel_indicbert_seed0")
    monkeypatch.setattr(checkpoints, "CHECKPOINT_ROOT", checkpoints_root)
    monkeypatch.setattr(predictions, "PREDICTIONS_ROOT", tmp_path / "predictions")

    condition = {
        "name": "test_partitioned_transfer_tel_to_ben",
        "kind": "transfer",
        "train": "task_3/T/tel",
        "eval": "task_3/B/ben",
        "eval_items": "partition:eval",
    }
    E.run_condition(
        3, condition, "task3_tel_indicbert_seed0", 0,
        device="cpu", n_boot=50, results_path=tmp_path / "experiments.csv",
        partition_seed=999,
    )

    log = predictions.read_prediction_log(3, "test_partitioned_transfer_tel_to_ben")
    full_split = load_native(3, "ben")
    assert 0 < len(log) < len(full_split)
    assert all(partition_of(i, seed=999) == "eval" for i in log["item_id"])


# --------------------------------------------------------------------------
# Encoder-addressable checkpoints (CLAUDE5.md T-500)
# --------------------------------------------------------------------------


def test_run_id_uses_the_encoder_slug_not_a_literal():
    """The T-500 blocker: `run_id_for` pasted 'indicbert' in literally, so
    `plan` reported every condition blocked for any other encoder."""
    assert E.run_id_for(2, "hin", 0, "mbert-base") == "task2_hin_mbert_seed0"
    assert E.run_id_for(3, "tel", 2, "xlm-r-base") == "task3_tel_xlmr_seed2"


def test_run_id_default_is_unchanged_for_indicbert():
    """Phase 2/3's committed checkpoints must keep resolving to their names."""
    assert E.run_id_for(2, "hin", 0) == "task2_hin_indicbert_seed0"


def test_run_id_rejects_an_unknown_encoder():
    with pytest.raises(ValueError, match="unknown encoder"):
        E.run_id_for(2, "hin", 0, "not-an-encoder")


def test_plan_names_the_requested_encoders_checkpoints():
    entries = E.plan(2, encoder="mbert-base")
    assert entries
    assert all(e["encoder"] == "mbert-base" for e in entries)
    assert all("_mbert_seed" in e["run_id"] for e in entries)


def test_shipped_config_stems_match_the_run_id_convention():
    """`checkpoints._current_hashes` maps a run_id back to
    `configs/train/<stem>.yaml`, so the convention has to hold in both
    directions or retention classifies every checkpoint as an orphan."""
    from src.config import TRAIN_CONFIG_DIR, load_run_config

    paths = sorted(TRAIN_CONFIG_DIR.glob("*.yaml")) + sorted(
        (TRAIN_CONFIG_DIR / "deferred").glob("*.yaml")
    )
    assert paths
    for path in paths:
        run = load_run_config(path)
        expected = E.run_id_for(run.task, run.lang, run.seed, run.encoder)
        assert expected == f"{path.stem}_seed{run.seed}", (
            f"{path.name} does not follow task{{n}}_{{lang}}_{{slug}}.yaml"
        )


# --------------------------------------------------------------------------
# One bad pair must not cost the whole sweep (§11, T-404)
# --------------------------------------------------------------------------


def test_run_all_reports_a_failed_pair_and_keeps_going(monkeypatch, tmp_path):
    """A dropped Hub connection three conditions into a 42-pair sweep used to
    abort the run and discard everything after it. transformers makes a live
    `model_info` request inside every uncached `from_pretrained`, so this is a
    transient that will recur — the sweep has to survive it."""
    from src import evaluate as E

    matrix = {
        "task": 2,
        "seed": 999,
        "conditions": [
            {"name": "c_ok_1", "kind": "transfer", "train": "task_2/H/hin",
             "eval": "task_2/B/ben", "eval_items": "all"},
            {"name": "c_boom", "kind": "transfer", "train": "task_2/H/hin",
             "eval": "task_2/T/tel", "eval_items": "all"},
            {"name": "c_ok_2", "kind": "transfer", "train": "task_2/H/hin",
             "eval": "task_2/B/mal", "eval_items": "all"},
        ],
    }
    monkeypatch.setattr(E, "load_matrix", lambda task: matrix)
    monkeypatch.setattr(
        E, "plan",
        lambda task, m=None, seeds=(0,), encoder=None: [
            {"condition": c["name"], "kind": c["kind"], "encoder": "indicbert-v2",
             "model_lang": "hin", "seed": 0, "run_id": "rid", "runnable": True}
            for c in matrix["conditions"]
        ],
    )

    calls = []

    def fake_run_condition(task, condition, run_id, seed, **kw):
        calls.append(condition["name"])
        if condition["name"] == "c_boom":
            raise ConnectionError("Remote end closed connection without response")
        return [{"condition": condition["name"]}]

    monkeypatch.setattr(E, "run_condition", fake_run_condition)

    result = E.run_all(2, results_path=tmp_path / "e.csv")

    assert calls == ["c_ok_1", "c_boom", "c_ok_2"], "the sweep must continue past the failure"
    assert len(result["ran"]) == 2
    assert len(result["failed"]) == 1
    assert result["failed"][0]["condition"] == "c_boom"
    assert "ConnectionError" in result["failed"][0]["error"]
    assert "Remote end closed" in result["failed"][0]["error"]


def test_run_all_still_separates_blocked_from_failed(monkeypatch, tmp_path):
    """Blocked means the checkpoint does not exist and the cell was never run;
    failed means it exists and broke. Collapsing them would make a missing
    Telugu baseline look like a bug and a real bug look like schedule."""
    from src import evaluate as E

    matrix = {"task": 2, "seed": 999, "conditions": [
        {"name": "c_blocked", "kind": "transfer", "train": "task_2/H/hin",
         "eval": "task_2/B/ben", "eval_items": "all"},
    ]}
    monkeypatch.setattr(E, "load_matrix", lambda task: matrix)
    monkeypatch.setattr(
        E, "plan",
        lambda task, m=None, seeds=(0,), encoder=None: [
            {"condition": "c_blocked", "kind": "transfer", "encoder": "indicbert-v2",
             "model_lang": "tel", "seed": 0, "run_id": "missing", "runnable": False}
        ],
    )
    result = E.run_all(2, results_path=tmp_path / "e.csv")
    assert len(result["blocked"]) == 1
    assert result["failed"] == []
    assert result["ran"] == []


def test_keyboard_interrupt_is_not_swallowed(monkeypatch, tmp_path):
    """Ctrl+C must still stop a long sweep immediately."""
    from src import evaluate as E

    matrix = {"task": 2, "seed": 999, "conditions": [
        {"name": "c", "kind": "transfer", "train": "task_2/H/hin",
         "eval": "task_2/B/ben", "eval_items": "all"},
    ]}
    monkeypatch.setattr(E, "load_matrix", lambda task: matrix)
    monkeypatch.setattr(
        E, "plan",
        lambda task, m=None, seeds=(0,), encoder=None: [
            {"condition": "c", "kind": "transfer", "encoder": "indicbert-v2",
             "model_lang": "hin", "seed": 0, "run_id": "rid", "runnable": True}
        ],
    )

    def interrupted(*a, **kw):
        raise KeyboardInterrupt

    monkeypatch.setattr(E, "run_condition", interrupted)
    with pytest.raises(KeyboardInterrupt):
        E.run_all(2, results_path=tmp_path / "e.csv")


def test_get_tokenizer_is_cached_per_encoder(monkeypatch):
    """Every uncached `from_pretrained` costs a live Hub request in
    transformers 4.57.6, and the sweep calls this once per condition arm."""
    from src import data as D

    D._load_tokenizer.cache_clear()
    loads = []

    class FakeAuto:
        @staticmethod
        def from_pretrained(hf_id):
            loads.append(hf_id)
            return object()

    import transformers

    monkeypatch.setattr(transformers, "AutoTokenizer", FakeAuto)
    first = D.get_tokenizer("indicbert-v2")
    second = D.get_tokenizer("indicbert-v2")
    third = D.get_tokenizer(D.resolve_encoder("indicbert-v2"))

    assert loads == ["ai4bharat/indic-bert"], f"loaded {len(loads)} times, expected 1"
    assert first is second is third
    D._load_tokenizer.cache_clear()
