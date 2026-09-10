"""Classification metrics for the training pipeline (CLAUDE2.md T-203 onward).

Macro-F1 is the primary number (`hard rule 6`): it weights every class equally,
so a model that ignores a minority class cannot hide behind accuracy. Accuracy
is reported next to it, never instead of it.

Kept separate from the training loop so T-206's run logging and T-207's gap
diagnosis compute the same numbers the same way.
"""

from __future__ import annotations

from collections.abc import Sequence

from sklearn.metrics import accuracy_score, f1_score


def classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_labels: int,
) -> dict:
    """Accuracy, macro-F1, and per-class F1 over a fixed label set.

    ``labels=range(num_labels)`` is passed explicitly so a class absent from a
    small dev fold still appears (as F1 0.0) rather than shifting the macro
    average onto whichever classes happened to show up.
    """
    labels = list(range(num_labels))
    per_class = f1_score(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
        ),
        "per_class_f1": [float(x) for x in per_class],
    }
