"""Classification metrics for the training and evaluation pipelines
(CLAUDE2.md T-203 onward; CLAUDE3.md T-303 adds the confusion matrix).

Macro-F1 is the primary number (`hard rule 6`): it weights every class equally,
so a model that ignores a minority class cannot hide behind accuracy. Accuracy
is reported next to it, never instead of it.

Kept separate from the training loop so T-206's run logging, T-207's gap
diagnosis, and Phase 3's zero-shot evaluation all compute the same numbers the
same way.
"""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_labels: int,
) -> dict:
    """Accuracy, macro-F1, per-class F1, and the confusion matrix over a fixed
    label set.

    ``labels=range(num_labels)`` is passed explicitly everywhere below so a
    class absent from a small split still occupies its row/column (as F1 0.0,
    an all-zero confusion-matrix row) rather than shifting the macro average
    or shrinking the matrix onto whichever classes happened to show up.

    ``confusion_matrix[i][j]`` is the count of items whose gold label is ``i``
    and whose predicted label is ``j`` -- sklearn's own convention (true rows,
    predicted columns), kept rather than transposed so it needs no translation
    when compared against `sklearn.metrics.confusion_matrix` directly.
    """
    labels = list(range(num_labels))
    per_class = f1_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "per_class_f1": [float(x) for x in per_class],
        "confusion_matrix": matrix.tolist(),
    }
