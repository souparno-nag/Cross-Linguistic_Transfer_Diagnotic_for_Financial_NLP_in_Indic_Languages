"""Tests for src/metrics.py.

The point of macro-F1 (`hard rule 6`) is that it does not let a model hide a
dead class behind accuracy, so the fixtures check exactly that gap.
"""

from src.metrics import classification_metrics


def test_perfect_prediction_scores_one():
    m = classification_metrics([0, 1, 2, 1, 0], [0, 1, 2, 1, 0], 3)
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["per_class_f1"] == [1.0, 1.0, 1.0]


def test_majority_guess_has_high_accuracy_but_low_macro_f1():
    # 8 of class 0, 2 of class 1; model always says 0.
    y_true = [0] * 8 + [1] * 2
    y_pred = [0] * 10
    m = classification_metrics(y_true, y_pred, 2)
    assert m["accuracy"] == 0.8
    assert m["macro_f1"] < 0.5
    assert m["per_class_f1"][1] == 0.0


def test_absent_class_still_occupies_a_slot():
    # class 2 never appears; per-class list must still be length 3.
    m = classification_metrics([0, 1, 0, 1], [0, 1, 0, 1], 3)
    assert len(m["per_class_f1"]) == 3
    assert m["per_class_f1"][2] == 0.0


def test_known_small_case():
    m = classification_metrics([0, 0, 0, 1], [0, 0, 0, 0], 2)
    assert m["accuracy"] == 0.75
    assert round(m["macro_f1"], 4) == 0.4286
