"""Tests for T-304's bootstrap CI and transfer-gap computation.

All fast -- this module is pure numpy/sklearn arithmetic, no model or corpus
access. The determinism tests are the done criterion itself (CLAUDE3.md
T-304: "reproducible under a fixed seed").
"""

from __future__ import annotations

import numpy as np
import pytest

from src import transfer as X


# --------------------------------------------------------------------------
# bootstrap_metric
# --------------------------------------------------------------------------


def test_same_seed_reproduces_identical_samples_and_ci():
    y_true = [0, 1, 2, 1, 0, 2, 1, 0, 2, 1]
    y_pred = [0, 1, 1, 1, 0, 2, 0, 0, 2, 1]
    a = X.bootstrap_metric(y_true, y_pred, 3, n_boot=200, seed=7)
    b = X.bootstrap_metric(y_true, y_pred, 3, n_boot=200, seed=7)
    assert a.samples == b.samples
    assert a.ci_low == b.ci_low and a.ci_high == b.ci_high
    assert a.point == b.point


def test_a_different_seed_gives_a_different_resample_distribution():
    y_true = [0, 1, 2, 1, 0, 2, 1, 0, 2, 1]
    y_pred = [0, 1, 1, 1, 0, 2, 0, 0, 2, 1]
    a = X.bootstrap_metric(y_true, y_pred, 3, n_boot=200, seed=1)
    b = X.bootstrap_metric(y_true, y_pred, 3, n_boot=200, seed=2)
    assert a.samples != b.samples


def test_point_estimate_matches_the_metric_on_the_full_set():
    from src.metrics import classification_metrics

    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]
    result = X.bootstrap_metric(y_true, y_pred, 3, n_boot=100, seed=0)
    expected = classification_metrics(y_true, y_pred, 3)["macro_f1"]
    assert result.point == pytest.approx(expected)


def test_perfect_prediction_has_a_degenerate_ci_at_one():
    # 20 items per class: a bootstrap resample missing a class entirely is
    # astronomically unlikely, so a perfect classifier's CI stays at 1.0. A
    # handful of items per class (tried first) does *not* hold this property
    # -- a resample can easily drop a whole class, and macro-F1 correctly
    # scores that as < 1.0 even though every resampled prediction was right;
    # that is the bootstrap finding something real, not a bug in it.
    y_true = [0, 1, 2] * 20
    y_pred = list(y_true)
    result = X.bootstrap_metric(y_true, y_pred, 3, n_boot=200, seed=0)
    assert result.point == 1.0
    assert result.ci_low == 1.0
    assert result.ci_high == 1.0


def test_a_small_sample_can_make_even_perfect_predictions_look_imperfect():
    """The bootstrap correctly penalises macro-F1 when a resample happens to
    drop a class entirely -- this is what makes small eval sets need a CI at
    all, not a quirk to work around."""
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = list(y_true)
    result = X.bootstrap_metric(y_true, y_pred, 3, n_boot=300, seed=0)
    assert result.point == 1.0
    assert result.ci_low < 1.0
    assert any(s < 1.0 for s in result.samples)


def test_ci_bounds_are_within_the_observed_sample_range():
    y_true = [0, 0, 0, 1, 1, 2, 2, 2, 1, 0]
    y_pred = [0, 1, 0, 1, 0, 2, 2, 0, 1, 0]
    result = X.bootstrap_metric(y_true, y_pred, 3, n_boot=500, seed=3)
    assert min(result.samples) <= result.ci_low <= result.ci_high <= max(result.samples)


def test_requested_n_boot_and_seed_are_recorded():
    result = X.bootstrap_metric([0, 1], [0, 1], 2, n_boot=50, seed=42)
    assert result.n_boot == 50
    assert result.seed == 42
    assert len(result.samples) == 50


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="vs"):
        X.bootstrap_metric([0, 1], [0], 2)


def test_empty_input_raises():
    with pytest.raises(ValueError, match="empty"):
        X.bootstrap_metric([], [], 2)


def test_invalid_ci_raises():
    with pytest.raises(ValueError, match="ci must be"):
        X.bootstrap_metric([0, 1], [0, 1], 2, ci=1.5)


def test_wider_ci_gives_a_wider_or_equal_interval():
    y_true = [0, 0, 0, 1, 1, 2, 2, 2, 1, 0]
    y_pred = [0, 1, 0, 1, 0, 2, 2, 0, 1, 0]
    narrow = X.bootstrap_metric(y_true, y_pred, 3, n_boot=500, seed=1, ci=0.80)
    wide = X.bootstrap_metric(y_true, y_pred, 3, n_boot=500, seed=1, ci=0.99)
    assert wide.ci_low <= narrow.ci_low
    assert wide.ci_high >= narrow.ci_high


# --------------------------------------------------------------------------
# transfer_gap
# --------------------------------------------------------------------------


def test_gap_is_source_point_minus_target_point():
    source = X.bootstrap_metric([0, 1, 1, 0, 1], [0, 1, 1, 0, 1], 2, n_boot=100, seed=0)
    target = X.bootstrap_metric([0, 1, 1, 0, 1], [0, 0, 0, 0, 0], 2, n_boot=100, seed=1)
    gap = X.transfer_gap(source, target)
    assert gap.gap == pytest.approx(source.point - target.point)
    assert gap.source == source.point
    assert gap.target == target.point


def test_gap_computation_is_deterministic_given_the_same_bootstrap_seeds():
    y_true = [0, 1, 1, 0, 1, 0, 1, 1]
    source = X.bootstrap_metric(y_true, y_true, 2, n_boot=300, seed=5)

    def make_target():
        y_pred = [1, 1, 0, 0, 1, 1, 0, 1]
        return X.bootstrap_metric(y_true, y_pred, 2, n_boot=300, seed=6)

    a = X.transfer_gap(source, make_target())
    b = X.transfer_gap(source, make_target())
    assert a == b


def test_mismatched_n_boot_is_refused():
    source = X.bootstrap_metric([0, 1], [0, 1], 2, n_boot=100, seed=0)
    target = X.bootstrap_metric([0, 1], [1, 0], 2, n_boot=200, seed=0)
    with pytest.raises(ValueError, match="must match"):
        X.transfer_gap(source, target)


def test_a_clear_gap_is_not_within_seed_noise():
    # source: perfect. target: consistently wrong. The gap should be large
    # and its CI should not straddle zero.
    y_true = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1] * 3
    source = X.bootstrap_metric(y_true, y_true, 2, n_boot=500, seed=0)
    y_pred_bad = [1 - y for y in y_true]
    target = X.bootstrap_metric(y_true, y_pred_bad, 2, n_boot=500, seed=1)
    gap = X.transfer_gap(source, target)
    assert gap.gap > 0.5
    assert not gap.within_seed_noise()


def test_an_identical_source_and_target_gap_is_within_seed_noise():
    y_true = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1] * 3
    y_pred = [0, 1, 0, 0, 1, 1, 1, 1, 0, 0] * 3
    source = X.bootstrap_metric(y_true, y_pred, 2, n_boot=500, seed=0)
    target = X.bootstrap_metric(y_true, y_pred, 2, n_boot=500, seed=1)
    gap = X.transfer_gap(source, target)
    assert gap.gap == pytest.approx(0.0, abs=1e-9)
    assert gap.within_seed_noise()


def test_to_dict_exposes_the_reportable_fields():
    source = X.bootstrap_metric([0, 1], [0, 1], 2, n_boot=50, seed=0)
    target = X.bootstrap_metric([0, 1], [1, 0], 2, n_boot=50, seed=1)
    gap = X.transfer_gap(source, target)
    assert set(gap.to_dict()) == {"source", "target", "gap", "gap_ci_low", "gap_ci_high"}
    assert set(source.to_dict()) == {"point", "ci_low", "ci_high", "n_boot", "seed"}
