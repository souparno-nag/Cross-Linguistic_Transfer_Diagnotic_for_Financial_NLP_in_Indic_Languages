"""Tests for T-308's results tables. Fast, hand-built fixtures -- prediction
logs and the eval-conditions matrix are both monkeypatched rather than read
from disk, so these run without a model, a checkpoint, or the corpus.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import results_tables as R


def _log(item_ids, gold, pred, *, condition_id, block="H", lang="ben", origin="mt", seed=0):
    n = len(item_ids)
    return pd.DataFrame(
        {
            "condition_id": [condition_id] * n,
            "run_id": [f"r_seed{seed}"] * n,
            "item_id": item_ids,
            "block_id": [block] * n,
            "lang": [lang] * n,
            "origin": [origin] * n,
            "src_lang": (["hin"] * n) if origin == "mt" else ([None] * n),
            "gold": gold,
            "pred": pred,
            "probs": [[0.5, 0.5]] * n,
            "seed": [seed] * n,
        }
    )


def _condition(name="transfer_hin_to_ben", quadrant="Indo-Aryan->Indo-Aryan", kind="transfer"):
    return {"name": name, "kind": kind, "quadrant": quadrant, "train": "task_2/H/hin", "eval": "task_2/B/ben"}


# --------------------------------------------------------------------------
# condition_seed_gap
# --------------------------------------------------------------------------


def test_condition_seed_gap_is_none_when_a_log_is_missing(monkeypatch):
    def fake_read(task, condition_id):
        raise FileNotFoundError(condition_id)

    monkeypatch.setattr(R, "read_prediction_log", fake_read)
    assert R.condition_seed_gap(2, _condition(), 0) is None


def test_condition_seed_gap_is_none_when_the_seed_has_no_rows(monkeypatch):
    source = _log(["a"], gold=[0], pred=[0], condition_id="H_hin_native_ceiling", block="H", lang="hin", origin="native", seed=1)
    target = _log(["a"], gold=[0], pred=[0], condition_id="transfer_hin_to_ben", seed=1)

    def fake_read(task, condition_id):
        return source if "ceiling" in condition_id else target

    monkeypatch.setattr(R, "read_prediction_log", fake_read)
    assert R.condition_seed_gap(2, _condition(), 0) is None  # asking for seed 0, only seed 1 exists


def test_condition_seed_gap_computes_source_minus_target(monkeypatch):
    # perfect source, always-wrong target -> gap should be large and positive
    items = [f"i{i}" for i in range(20)]
    gold = [i % 2 for i in range(20)]
    source = _log(items, gold=gold, pred=gold, condition_id="H_hin_native_ceiling", block="H", lang="hin", origin="native", seed=0)
    target = _log(items, gold=gold, pred=[1 - g for g in gold], condition_id="transfer_hin_to_ben", seed=0)

    def fake_read(task, condition_id):
        return source if "ceiling" in condition_id else target

    monkeypatch.setattr(R, "read_prediction_log", fake_read)
    result = R.condition_seed_gap(2, _condition(), 0, n_boot=100)
    assert result["n_items"] == 20
    assert result["source"] == 1.0
    assert result["target"] == 0.0
    assert result["gap"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# condition_summary
# --------------------------------------------------------------------------


def test_condition_summary_is_blocked_when_no_seed_has_data(monkeypatch):
    monkeypatch.setattr(R, "read_prediction_log", lambda *a: (_ for _ in ()).throw(FileNotFoundError()))
    summary = R.condition_summary(2, _condition())
    assert summary["status"] == "blocked"
    assert summary["n_seeds"] == 0
    assert summary["condition"] == "transfer_hin_to_ben"


def test_condition_summary_aggregates_across_seeds(monkeypatch):
    items = [f"i{i}" for i in range(20)]
    gold = [i % 2 for i in range(20)]

    # read_prediction_log returns the *whole* log (all seeds); condition_seed_gap
    # filters to one seed itself -- so build one combined frame per
    # condition_id covering all three seeds.
    ceiling = pd.concat(
        [_log(items, gold=gold, pred=gold, condition_id="H_hin_native_ceiling", block="H", lang="hin", origin="native", seed=s) for s in (0, 1, 2)],
        ignore_index=True,
    )

    def make_target_seed(seed):
        wrong_count = {0: 0, 1: 5, 2: 10}[seed]
        pred = list(gold)
        for i in range(wrong_count):
            pred[i] = 1 - pred[i]
        return _log(items, gold=gold, pred=pred, condition_id="transfer_hin_to_ben", seed=seed)

    target = pd.concat([make_target_seed(s) for s in (0, 1, 2)], ignore_index=True)

    def fake_read2(task, condition_id):
        return ceiling if "ceiling" in condition_id else target

    monkeypatch.setattr(R, "read_prediction_log", fake_read2)
    summary = R.condition_summary(2, _condition(), n_boot=100)
    assert summary["status"] == "ok"
    assert summary["n_seeds"] == 3

    # Expected gap per seed: source is perfect (macro-F1 1.0); target's
    # macro-F1 is computed the same way the module computes it (not assumed
    # to equal 1 - wrong_count/n, which does not hold exactly for macro-F1
    # under an uneven per-class error split).
    from src.metrics import classification_metrics

    expected_gaps = []
    for seed in (0, 1, 2):
        wrong_count = {0: 0, 1: 5, 2: 10}[seed]
        pred = list(gold)
        for i in range(wrong_count):
            pred[i] = 1 - pred[i]
        expected_gaps.append(1.0 - classification_metrics(gold, pred, 2)["macro_f1"])

    import statistics as _stats

    assert summary["gap_mean"] == pytest.approx(_stats.fmean(expected_gaps), abs=1e-6)
    assert summary["gap_std"] > 0
    assert summary["within_seed_noise"] in (True, False)


# --------------------------------------------------------------------------
# condition_matrix / quadrant_summary
# --------------------------------------------------------------------------


def test_condition_matrix_keeps_blocked_cells_rather_than_dropping_them(monkeypatch):
    matrix = {
        "conditions": [
            _condition("transfer_hin_to_ben", "Indo-Aryan->Indo-Aryan"),
            _condition("transfer_ben_to_hin", "Indo-Aryan->Indo-Aryan", kind="transfer"),
        ]
    }
    monkeypatch.setattr(R, "load_matrix", lambda task: matrix)
    monkeypatch.setattr(R, "read_prediction_log", lambda *a: (_ for _ in ()).throw(FileNotFoundError()))
    df = R.condition_matrix(2, "transfer")
    assert len(df) == 2
    assert set(df["status"]) == {"blocked"}


def test_quadrant_summary_only_averages_ok_cells(monkeypatch):
    items = [f"i{i}" for i in range(10)]
    gold = [0] * 10
    ceiling = _log(items, gold=gold, pred=gold, condition_id="H_hin_native_ceiling", block="H", lang="hin", origin="native", seed=0)
    good_target = _log(items, gold=gold, pred=gold, condition_id="transfer_hin_to_ben", seed=0)  # gap 0

    matrix = {
        "conditions": [
            _condition("transfer_hin_to_ben", "Indo-Aryan->Indo-Aryan"),
            _condition("transfer_ben_to_hin", "Indo-Aryan->Indo-Aryan"),  # will be blocked
        ]
    }
    monkeypatch.setattr(R, "load_matrix", lambda task: matrix)

    def fake_read(task, condition_id):
        if condition_id == "transfer_ben_to_hin" or condition_id == "B_ben_native_ceiling":
            raise FileNotFoundError(condition_id)
        return ceiling if "ceiling" in condition_id else good_target

    monkeypatch.setattr(R, "read_prediction_log", fake_read)
    summary = R.quadrant_summary(2, "transfer", n_boot=50)
    ia_ia = summary[summary["quadrant"] == "Indo-Aryan->Indo-Aryan"].iloc[0]
    assert ia_ia["n_cells"] == 1
    assert ia_ia["gap_mean"] == pytest.approx(0.0)
    other = summary[summary["quadrant"] == "Indo-Aryan->Dravidian"].iloc[0]
    assert other["n_cells"] == 0
    assert pd.isna(other["gap_mean"])  # None becomes NaN once in a float column


# --------------------------------------------------------------------------
# translationese_comparison
# --------------------------------------------------------------------------


def test_translationese_comparison_reports_delta_from_native(monkeypatch):
    items = [f"i{i}" for i in range(10)]
    gold = [i % 2 for i in range(10)]
    native = _log(items, gold=gold, pred=gold, condition_id="translationese_hin", block="H", lang="hin", origin="native", seed=0)
    from_ben = _log(items, gold=gold, pred=[1 - g for g in gold], condition_id="translationese_hin", block="B", lang="hin", origin="mt", seed=0)
    log = pd.concat([native, from_ben], ignore_index=True)

    monkeypatch.setattr(R, "read_prediction_log", lambda task, cid: log)
    df = R.translationese_comparison(2, "hin", seeds=(0,))
    native_row = df[df["origin"] == "native"].iloc[0]
    ben_row = df[df["block"] == "B"].iloc[0]
    assert native_row["macro_f1_mean"] == pytest.approx(1.0)
    assert ben_row["macro_f1_mean"] == pytest.approx(0.0)
    assert ben_row["delta_vs_native"] == pytest.approx(-1.0)


def test_translationese_comparison_returns_empty_frame_when_missing(monkeypatch):
    monkeypatch.setattr(R, "read_prediction_log", lambda *a: (_ for _ in ()).throw(FileNotFoundError()))
    df = R.translationese_comparison(2, "ben")
    assert df.empty


# --------------------------------------------------------------------------
# Markdown rendering
# --------------------------------------------------------------------------


def test_is_missing_treats_none_and_nan_the_same():
    assert R.is_missing(None)
    assert R.is_missing(float("nan"))
    assert not R.is_missing(0.0)
    assert not R.is_missing(False)  # a real boolean, not a missing value


def test_render_matrix_table_shows_a_dash_when_every_within_seed_noise_is_none():
    """Regression: when every 'ok' row's within_seed_noise is None (only one
    seed available so far, e.g.), pandas coerces that whole column to NaN --
    and bool(float("nan")) is True in Python, so a naive `is None` check
    rendered every genuinely-unknown value as "yes"."""
    df = pd.DataFrame(
        [
            {
                "condition": "transfer_hin_to_ben", "quadrant": "Indo-Aryan->Indo-Aryan", "status": "ok",
                "n_seeds": 1, "source_mean": 0.9, "target_mean": 0.6, "gap_mean": 0.3, "gap_std": 0.0,
                "within_seed_noise": None,
            }
        ]
    )
    assert df["within_seed_noise"].isna().all()  # confirms pandas did coerce it to NaN
    lines = R.render_matrix_table(df, title="t")
    row_line = next(l for l in lines if l.startswith("| transfer_hin_to_ben"))
    assert row_line.endswith("| — |")  # not "| yes |"


def test_render_matrix_table_reports_a_real_within_seed_noise_value():
    df = pd.DataFrame(
        [
            {
                "condition": "c", "quadrant": "q", "status": "ok", "n_seeds": 3,
                "source_mean": 0.9, "target_mean": 0.6, "gap_mean": 0.3, "gap_std": 0.5,
                "within_seed_noise": True,
            },
            {
                "condition": "c2", "quadrant": "q", "status": "ok", "n_seeds": 3,
                "source_mean": 0.9, "target_mean": 0.2, "gap_mean": 0.7, "gap_std": 0.1,
                "within_seed_noise": False,
            },
        ]
    )
    lines = R.render_matrix_table(df, title="t")
    assert next(l for l in lines if l.startswith("| c |")).endswith("| yes |")
    assert next(l for l in lines if l.startswith("| c2 |")).endswith("| no |")


def test_render_matrix_table_renders_blocked_rows():
    df = pd.DataFrame([{"condition": "transfer_ben_to_hin", "quadrant": "Indo-Aryan->Indo-Aryan", "status": "blocked", "n_seeds": 0}])
    lines = R.render_matrix_table(df, title="t")
    assert any("blocked" in l for l in lines)
