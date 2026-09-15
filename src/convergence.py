"""Did a training run actually train? (CLAUDE6.md T-402.)

A fine-tuning run can finish, log a number, and land in the baseline table
without ever having left its initialisation. That happened once in Phase 4:
`task3_ben_indicbert_seed0` early-stopped at epoch 13 with a peak **train**
macro-F1 of 0.152, while its two sibling seeds — same data, same config,
different seed only — reached 0.688 and 0.997. Its reported test macro-F1 of
0.064 is therefore not a measurement of Bengali task-3 difficulty; it is a
measurement of a run that never started learning, and averaging it into the
baseline moved the mean by 3 points and quadrupled the standard deviation.

This module answers one narrow question — *did the model fit the data it was
shown?* — so that such a run can be reported **alongside** the full average
rather than silently dropped from it (`hard rule 3`: never drop a run, and
never tune a threshold to tidy a distribution).

Why the training fold is the right place to look
------------------------------------------------

Test macro-F1 cannot distinguish the two failures that matter here. Task 3 has
532 rows over 10 classes, so a *healthy* run scores about 0.15 on test — the
data ceiling CLAUDE2.md T-206/T-207 diagnosed and the published IndicFinNLP
baseline confirms. A run that never trained also scores about that. The two are
indistinguishable from the outside.

On the **training** fold they are not remotely alike. These configs are
deliberately budgeted long enough to memorise: T-206 records task 3 reaching
train macro-F1 0.99 while dev sits at 0.15, and diagnoses it as "overfits
hard". Fitting the training set is the expected behaviour of a completed run
here, not a warning sign, which makes failure to fit it an unambiguous signal
that the optimisation did not get going.

The threshold, and why it is not a knife edge
---------------------------------------------

`MIN_TRAIN_MACRO_F1 = 0.5` asks only that a model separate classes on data it
has seen ten to forty times. It is set deliberately far below what a healthy
run reaches, and it is a judgement, so it is stated rather than buried. The
observed populations are nowhere near it — every completed run in the project
sits at 0.52 or above and most are above 0.97, while the one failed run sits at
0.15:

    0.15  |  task3_ben_indicbert_seed0        <- did not train
    ------|---- 0.5 threshold ----
    0.52  |  task3_hin_indicbert_seed2
    0.69  |  task3_ben_indicbert_seed2
    0.97  |  task3_hin_indicbert_seed1
    0.99+ |  everything else, both tasks

Moving the threshold anywhere between 0.16 and 0.51 classifies every run in the
project identically, so the result does not depend on the exact value. That is
the property that makes it reportable; if a future run lands near 0.5, the
threshold has stopped being informative and the run needs reading by hand
rather than a nudged constant.

`None` means *unknown*, not *failed*
------------------------------------

Convergence is read from the run's checkpoint history, and checkpoints are
gitignored and prunable (T-208). A run whose checkpoint is gone returns `None`
and is **kept** in the converged subset, because the subset exists to exclude
runs with positive evidence of failure, not runs we cannot assess. Excluding
the unknown would let deleting a checkpoint quietly change a published number.
"""

from __future__ import annotations

from pathlib import Path

MIN_TRAIN_MACRO_F1 = 0.5


def peak_train_macro_f1(history: list[dict] | None) -> float | None:
    """The best train macro-F1 over a run's epochs — `None` if unknown.

    The *peak* rather than the final value, because early stopping restores
    the best-dev weights and a run can end on a worse epoch than its best
    without that meaning it failed to fit.
    """
    if not history:
        return None
    values = [
        float(epoch["train_macro_f1"])
        for epoch in history
        if epoch.get("train_macro_f1") is not None
    ]
    return max(values) if values else None


def converged(history: list[dict] | None) -> bool | None:
    """Did this run fit its training data? `None` when it cannot be assessed."""
    peak = peak_train_macro_f1(history)
    if peak is None:
        return None
    return peak >= MIN_TRAIN_MACRO_F1


def history_for(run_id: str, root: str | Path | None = None) -> list[dict] | None:
    """A run's epoch history from its checkpoint, or `None` if unavailable.

    Never raises: a missing, pruned or unreadable checkpoint is an *unknown*
    convergence state, and the report says so rather than failing the sweep
    over a file that is regenerable by design.
    """
    from . import checkpoints

    try:
        path = (
            checkpoints.checkpoint_path(run_id)
            if root is None
            else Path(root) / run_id / "checkpoint.pt"
        )
        if not Path(path).exists():
            return None
        import torch

        blob = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    history = blob.get("history")
    return history if isinstance(history, list) else None


def summarise_convergence(records: list[dict]) -> dict:
    """Split a config's runs into those that trained and those that did not.

    `records` are the per-seed report rows; each may carry `converged`. Returns
    the two seed lists plus the subset used for the alongside figure.
    """
    failed = sorted(r["seed"] for r in records if r.get("converged") is False)
    unknown = sorted(r["seed"] for r in records if r.get("converged") is None)
    kept = [r for r in records if r.get("converged") is not False]
    return {
        "n_converged": len(kept),
        "nonconverged_seeds": failed,
        "unknown_seeds": unknown,
        "converged_records": kept,
    }
