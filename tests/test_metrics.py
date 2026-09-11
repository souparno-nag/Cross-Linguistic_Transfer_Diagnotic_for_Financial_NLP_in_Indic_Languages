"""Tests for src/metrics.py.

The point of macro-F1 (`hard rule 6`) is that it does not let a model hide a
dead class behind accuracy, so the fixtures check exactly that gap. The
confusion-matrix tests (CLAUDE3.md T-303) validate against
`sklearn.metrics.confusion_matrix` directly on a fixture worked out by hand,
rather than trusting that calling sklearn internally is validation on its own.
"""

import numpy as np
from sklearn.metrics import confusion_matrix as sklearn_confusion_matrix

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


# --------------------------------------------------------------------------
# T-303: confusion matrix, validated against sklearn on a hand-checked fixture
# --------------------------------------------------------------------------


def test_confusion_matrix_matches_sklearn_directly():
    y_true = [0, 0, 1, 1, 2, 2, 2, 0]
    y_pred = [0, 1, 1, 1, 2, 0, 2, 0]
    m = classification_metrics(y_true, y_pred, 3)
    expected = sklearn_confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    assert np.array_equal(np.array(m["confusion_matrix"]), expected)


def test_confusion_matrix_hand_worked_example():
    # 3 true-0 (2 correct, 1 called 1); 2 true-1 (both correct); 3 true-2
    # (2 correct, 1 called 0). Row i = true label i, column j = predicted j.
    y_true = [0, 0, 0, 1, 1, 2, 2, 2]
    y_pred = [0, 0, 1, 1, 1, 2, 2, 0]
    m = classification_metrics(y_true, y_pred, 3)
    assert m["confusion_matrix"] == [
        [2, 1, 0],
        [0, 2, 0],
        [1, 0, 2],
    ]


def test_confusion_matrix_gives_an_absent_class_an_all_zero_row_and_column():
    # class 2 never appears in this fixture, but must still occupy a slot
    # (same reasoning as per_class_f1's absent-class test above).
    m = classification_metrics([0, 1, 0, 1], [0, 1, 0, 1], 3)
    matrix = m["confusion_matrix"]
    assert len(matrix) == 3 and all(len(row) == 3 for row in matrix)
    assert matrix[2] == [0, 0, 0]
    assert [row[2] for row in matrix] == [0, 0, 0]


def test_confusion_matrix_diagonal_sums_to_correct_count():
    y_true = [0, 1, 2, 0, 1, 2, 0]
    y_pred = [0, 1, 1, 0, 2, 2, 1]
    m = classification_metrics(y_true, y_pred, 3)
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    diagonal = sum(m["confusion_matrix"][i][i] for i in range(3))
    assert diagonal == correct == 4
