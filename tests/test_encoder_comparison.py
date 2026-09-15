"""Tests for T-504's cross-encoder comparison (CLAUDE5.md).

The per-encoder numbers come from T-304/T-306 and are tested there. What is
tested here is what this module adds: aggregating per-seed confidence
intervals, differencing two encoders on the same condition, and keeping
blocked cells and excluded encoders visible instead of dropping them.
"""

import pandas as pd
import pytest

from src import encoder_comparison as C


# --------------------------------------------------------------------------
# Is a difference bigger than the encoders' own spread?
# --------------------------------------------------------------------------


def test_delta_larger_than_combined_spread_counts():
    # 0.20 against sqrt(0.01^2 + 0.02^2) ~= 0.022
    assert C.exceeds_seed_noise(0.20, 0.01, 0.02) is True


def test_delta_inside_combined_spread_does_not_count():
    assert C.exceeds_seed_noise(0.01, 0.03, 0.04) is False


def test_noise_check_is_symmetric_in_sign():
    assert C.exceeds_seed_noise(-0.20, 0.01, 0.02) is True
    assert C.exceeds_seed_noise(0.20, 0.01, 0.02) is True


def test_noise_check_combines_both_encoders_not_just_one():
    """A delta that clears one encoder's spread but not the pair's must not
    be reported as real."""
    assert C.exceeds_seed_noise(0.05, 0.01, 0.06) is False


def test_noise_check_is_none_when_anything_is_missing():
    assert C.exceeds_seed_noise(None, 0.01, 0.02) is None
    assert C.exceeds_seed_noise(0.2, None, 0.02) is None
    assert C.exceeds_seed_noise(float("nan"), 0.01, 0.02) is None


# --------------------------------------------------------------------------
# Aggregating seeds, keeping the intervals
# --------------------------------------------------------------------------


def _seed_gap(gap, lo, hi, *, source=0.9, target=0.4, n=100):
    return {
        "n_items": n, "source": source, "target": target,
        "gap": gap, "gap_ci_low": lo, "gap_ci_high": hi,
    }


def test_summary_averages_the_per_seed_intervals(monkeypatch):
    """T-304 bootstraps each seed separately, so three intervals exist and the
    summary reports their mean — not a single pooled interval, which would be
    narrower and unearned."""
    gaps = [_seed_gap(0.5, 0.4, 0.6), _seed_gap(0.6, 0.5, 0.7), _seed_gap(0.4, 0.3, 0.5)]
    monkeypatch.setattr(C, "condition_seed_gap", lambda *a, **k: gaps.pop(0))
    row = C.summarise_condition(2, {"name": "c", "kind": "transfer"}, "mbert-base")
    assert row["status"] == "ok"
    assert row["gap_mean"] == pytest.approx(0.5)
    assert row["gap_ci_low"] == pytest.approx(0.4)
    assert row["gap_ci_high"] == pytest.approx(0.6)
    assert row["n_seeds"] == 3


def test_summary_is_blocked_when_no_seed_has_data(monkeypatch):
    monkeypatch.setattr(C, "condition_seed_gap", lambda *a, **k: None)
    row = C.summarise_condition(2, {"name": "c", "kind": "transfer"}, "mbert-base")
    assert row["status"] == "blocked"
    assert row["n_seeds"] == 0
    assert row["encoder"] == "mbert-base"


def test_summary_carries_the_encoder_so_rows_cannot_be_confused(monkeypatch):
    monkeypatch.setattr(C, "condition_seed_gap", lambda *a, **k: _seed_gap(0.5, 0.4, 0.6))
    a = C.summarise_condition(2, {"name": "c", "kind": "transfer"}, "indicbert-v2")
    b = C.summarise_condition(2, {"name": "c", "kind": "transfer"}, "mbert-base")
    assert a["encoder"] != b["encoder"]


# --------------------------------------------------------------------------
# The comparison pivot
# --------------------------------------------------------------------------


def _row(encoder, *, condition="c", status="ok", target=0.4, target_std=0.01,
         gap=0.5, gap_std=0.02, task=2):
    return {
        "task": task, "condition": condition, "kind": "transfer",
        "quadrant": "Indo-Aryan->Indo-Aryan", "encoder": encoder, "status": status,
        "n_seeds": 3 if status == "ok" else 0, "target_mean": target,
        "target_std": target_std, "gap_mean": gap, "gap_std": gap_std,
        "gap_ci_low": gap - 0.05, "gap_ci_high": gap + 0.05,
    }


def test_compare_reports_the_delta_against_the_baseline_encoder():
    rows = pd.DataFrame([
        _row("indicbert-v2", target=0.30),
        _row("mbert-base", target=0.45),
    ])
    wide = C.compare(rows, baseline="indicbert-v2")
    assert len(wide) == 1
    assert wide.iloc[0]["mbert-base__target_delta"] == pytest.approx(0.15)
    # No delta column for the baseline against itself.
    assert "indicbert-v2__target_delta" not in wide.columns


def test_compare_leaves_the_delta_missing_when_either_side_is_blocked():
    rows = pd.DataFrame([
        _row("indicbert-v2", status="blocked"),
        _row("mbert-base"),
    ])
    wide = C.compare(rows, baseline="indicbert-v2")
    assert wide.iloc[0]["mbert-base__target_delta"] is None
    assert wide.iloc[0]["mbert-base__beats_noise"] is None
    # The blocked cell is still a row, not dropped.
    assert wide.iloc[0]["indicbert-v2__status"] == "blocked"


def test_compare_keeps_every_condition_including_fully_blocked_ones():
    rows = pd.DataFrame([
        _row("indicbert-v2", condition="a"),
        _row("mbert-base", condition="a"),
        _row("indicbert-v2", condition="b", status="blocked"),
        _row("mbert-base", condition="b", status="blocked"),
    ])
    wide = C.compare(rows, baseline="indicbert-v2")
    assert sorted(wide["condition"]) == ["a", "b"]


def test_compare_separates_the_same_condition_name_across_tasks():
    """task 2 and task 3 both have a condition called transfer_hin_to_ben."""
    rows = pd.DataFrame([
        _row("indicbert-v2", task=2, target=0.8), _row("mbert-base", task=2, target=0.9),
        _row("indicbert-v2", task=3, target=0.1), _row("mbert-base", task=3, target=0.3),
    ])
    wide = C.compare(rows, baseline="indicbert-v2")
    assert len(wide) == 2
    by_task = {int(r["task"]): r for _, r in wide.iterrows()}
    assert by_task[2]["mbert-base__target_delta"] == pytest.approx(0.1)
    assert by_task[3]["mbert-base__target_delta"] == pytest.approx(0.2)


def test_compare_on_empty_input_returns_an_empty_frame():
    assert C.compare(pd.DataFrame()).empty


# --------------------------------------------------------------------------
# Rendering: nothing quietly disappears
# --------------------------------------------------------------------------


def test_excluded_encoder_appears_in_the_baseline_table_with_its_reason():
    """CLAUDE5.md asks for all three encoders. XLM-R was excluded for a
    measured reason, and dropping the row would hide the exclusion."""
    frame = pd.DataFrame([{
        "task": 2, "config": "task2_hin_indicbert.yaml", "encoder": "indicbert-v2",
        "n_seeds": 3, "macro_f1_mean": 0.82, "macro_f1_std": 0.004,
        "batch_size": 16, "grad_accum": 1,
    }])
    md = "\n".join(C.render_baselines(frame))
    assert "xlm-r-base" in md
    assert "excluded" in md
    assert "4.14 GiB" in md


def test_blocked_cells_render_as_blocked_rather_than_vanishing():
    rows = pd.DataFrame([
        _row("indicbert-v2", condition="a"),
        _row("mbert-base", condition="a"),
        _row("indicbert-v2", condition="b", status="blocked"),
        _row("mbert-base", condition="b", status="blocked"),
    ])
    wide = C.compare(rows, baseline="indicbert-v2")
    md = "\n".join(C.render_comparison(wide, kind="transfer", task=2))
    assert "| a |" in md and "| b |" in md
    assert "blocked" in md


def test_render_shows_the_interval_beside_the_gap():
    rows = pd.DataFrame([_row("indicbert-v2"), _row("mbert-base")])
    wide = C.compare(rows, baseline="indicbert-v2")
    md = "\n".join(C.render_comparison(wide, kind="transfer", task=2))
    assert "[0.450, 0.550]" in md


def test_render_signs_the_delta():
    rows = pd.DataFrame([
        _row("indicbert-v2", target=0.30), _row("mbert-base", target=0.45),
    ])
    md = "\n".join(
        C.render_comparison(C.compare(rows, baseline="indicbert-v2"), kind="transfer", task=2)
    )
    assert "+0.1500" in md
