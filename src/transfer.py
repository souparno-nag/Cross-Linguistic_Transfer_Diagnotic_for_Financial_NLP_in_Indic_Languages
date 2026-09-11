"""Bootstrap confidence intervals and the transfer gap (CLAUDE3.md T-304).

Every zero-shot number Phase 3 reports needs an error bar: a transfer gap
smaller than its own sampling noise is not a finding (hard rule 5), and the
only way to tell the difference is to measure the noise. This module measures
it by the standard non-parametric bootstrap -- resample items with
replacement, recompute the metric, repeat -- rather than assuming a
parametric distribution classification metrics do not have.

The transfer gap itself is XTREME's convention (Hu et al., 2020): in-language
(source) score minus cross-lingual (target) score. A positive gap is
performance lost by moving languages; the sign is XTREME's, not one this
module invents.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .metrics import classification_metrics

DEFAULT_N_BOOT = 1000
DEFAULT_SEED = 0
DEFAULT_CI = 0.95


@dataclass(frozen=True)
class BootstrapResult:
    """One score's point estimate, its bootstrap CI, and the resampled
    distribution both the CI and `transfer_gap` are computed from."""

    point: float
    ci_low: float
    ci_high: float
    n_boot: int
    seed: int
    samples: tuple[float, ...]

    def to_dict(self) -> dict:
        return {
            "point": self.point,
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "n_boot": self.n_boot,
            "seed": self.seed,
        }


def bootstrap_metric(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_labels: int,
    *,
    metric: str = "macro_f1",
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    ci: float = DEFAULT_CI,
) -> BootstrapResult:
    """The non-parametric bootstrap CI for one classification metric.

    Resamples item *indices* with replacement -- keeping each item's
    ``(true, pred)`` pair together, since resampling them independently would
    measure something other than this model's accuracy on real items --
    recomputes ``metric`` on each of `n_boot` resamples, and reports the
    ``((1-ci)/2, 1-(1-ci)/2)`` percentiles as the interval. Deterministic
    (§4 rule 7): the same `seed` reproduces the same resamples, every time,
    on this or any other machine, because `numpy.random.default_rng` is
    seeded explicitly rather than left to ambient global state.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) != len(y_pred):
        raise ValueError(f"{len(y_true)} true labels vs {len(y_pred)} predictions")
    if len(y_true) == 0:
        raise ValueError("cannot bootstrap an empty set of predictions")
    if not 0.0 < ci < 1.0:
        raise ValueError(f"ci must be in (0, 1), got {ci}")

    point = classification_metrics(y_true, y_pred, num_labels)[metric]

    rng = np.random.default_rng(seed)
    n = len(y_true)
    samples = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        samples[i] = classification_metrics(y_true[idx], y_pred[idx], num_labels)[metric]

    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(samples, [alpha, 1.0 - alpha])
    return BootstrapResult(
        point=float(point),
        ci_low=float(lo),
        ci_high=float(hi),
        n_boot=n_boot,
        seed=seed,
        samples=tuple(float(x) for x in samples),
    )


@dataclass(frozen=True)
class TransferGap:
    """``source - target`` (XTREME's convention), with its own bootstrap CI."""

    source: float
    target: float
    gap: float
    gap_ci_low: float
    gap_ci_high: float

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "target": self.target,
            "gap": self.gap,
            "gap_ci_low": self.gap_ci_low,
            "gap_ci_high": self.gap_ci_high,
        }

    def within_seed_noise(self) -> bool:
        """Hard rule 5: a gap whose CI straddles zero is not a finding --
        the model could just as easily have scored higher on the target than
        the source on a re-sample, so "transfer hurt performance" is not
        supported yet."""
        return self.gap_ci_low <= 0.0 <= self.gap_ci_high


def transfer_gap(
    source: BootstrapResult, target: BootstrapResult, *, ci: float = DEFAULT_CI
) -> TransferGap:
    """The gap between two already-bootstrapped scores, with its own CI.

    Source and target are, in general, entirely different rows -- e.g. Hindi
    in-language text versus Bengali evaluation text -- so there is no shared
    item to pair a resample on. The CI is instead built the standard way for
    the difference of two *independent* distributions: pairing the two
    resampled-metric arrays position-for-position (``source.samples[i] -
    target.samples[i]``) is a valid draw from the difference's distribution
    for every ``i``, because each side's draws are independent of the other's
    regardless of which index they landed at. This requires equal `n_boot` on
    both sides so the arrays pair up one-to-one.
    """
    if source.n_boot != target.n_boot:
        raise ValueError(
            f"source has {source.n_boot} bootstrap draws, target has "
            f"{target.n_boot} -- both must match to pair the distributions"
        )
    diffs = np.asarray(source.samples) - np.asarray(target.samples)
    alpha = (1.0 - ci) / 2.0
    lo, hi = np.quantile(diffs, [alpha, 1.0 - alpha])
    return TransferGap(
        source=source.point,
        target=target.point,
        gap=source.point - target.point,
        gap_ci_low=float(lo),
        gap_ci_high=float(hi),
    )
