"""Did a run actually train? (CLAUDE6.md T-402, src/convergence.py.)

The failure this guards against is quiet by construction. On task 3 — 532 rows
over 10 classes — a healthy run scores about 0.15 on test, and a run that never
left its initialisation also scores about 0.15 on test. The two are
indistinguishable from the outside, which is why the check looks at the
*training* fold, where a completed run reaches 0.99 and a failed one stays near
the majority-class floor.

These tests pin three things: the criterion fires on the shape of a failed run,
it does not fire on a merely weak one, and an unassessable run is never
silently treated as failed.
"""

from __future__ import annotations

import pytest

from src import convergence as C


def _history(train_f1s):
    return [
        {
            "epoch": i,
            "train_loss": 2.3 - 0.1 * i,
            "train_macro_f1": f,
            "dev_macro_f1": f / 2,
            "dev_accuracy": 0.2,
        }
        for i, f in enumerate(train_f1s)
    ]


# --------------------------------------------------------------------------
# The criterion
# --------------------------------------------------------------------------


def test_a_run_that_memorised_its_training_set_converged():
    assert C.converged(_history([0.03, 0.4, 0.9, 0.99])) is True


def test_a_run_stuck_at_the_majority_class_floor_did_not():
    """`task3_ben_indicbert_seed0`'s actual shape: a brief excursion to 0.15,
    then back to the floor, then early-stopped."""
    history = _history([0.029, 0.029, 0.068, 0.056, 0.078, 0.152, 0.029, 0.029])
    assert C.converged(history) is False
    assert C.peak_train_macro_f1(history) == pytest.approx(0.152)


def test_peak_is_used_not_the_final_epoch():
    """Early stopping restores best-dev weights, so a run can end on a worse
    epoch than its best without having failed to fit."""
    assert C.converged(_history([0.1, 0.99, 0.6])) is True
    assert C.peak_train_macro_f1(_history([0.1, 0.99, 0.6])) == pytest.approx(0.99)


def test_a_weak_but_real_run_is_not_flagged():
    """Hindi task-3 seed 2 peaked at 0.5154 — poor, but it trained. Flagging
    it would turn the check into a quality filter, which is not what it is."""
    assert C.converged(_history([0.03, 0.3, 0.5154])) is True


def test_threshold_is_not_a_knife_edge_for_the_real_runs():
    """Every run on disk should sit well clear of the threshold on one side or
    the other, so the verdict does not depend on the exact value.

    **This reads the actual checkpoints.** An earlier version asserted against
    a hardcoded list of the peaks observed when it was written, which made it
    incapable of ever firing — and it duly stayed green when
    `task3_tel_indicbert_seed0` landed at 0.4692, right in the middle of the
    band it was supposed to be watching. A guard that cannot observe new data
    is not a guard.

    Skips when no checkpoints are present: `checkpoints/` is gitignored and
    prunable (T-208), so their absence is "nothing to check", not a failure.

    When this fails, do **not** move `MIN_TRAIN_MACRO_F1` to make it pass —
    that is `hard rule 3` exactly. Read the run by hand and report what it
    shows.
    """
    from pathlib import Path

    from src.download_dataset.paths import REPO_ROOT

    root = REPO_ROOT / "checkpoints"
    if not root.exists():
        pytest.skip("no checkpoints on disk")

    # Smoke checkpoints are deliberately one epoch on a 200-row subset
    # (T-401/T-500), so "did it converge" is not a meaningful question for
    # them. Excluding them scopes the guard to runs where convergence is a
    # claim; it does not move the threshold.
    checked, ambiguous = 0, []
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if run_dir.name.startswith(("t500_smoke__", "t401_smoke__")):
            continue
        if not (run_dir / "checkpoint.pt").exists():
            continue
        peak = C.peak_train_macro_f1(C.history_for(run_dir.name))
        if peak is None:
            continue
        checked += 1
        if 0.16 < peak < 0.51:
            ambiguous.append((run_dir.name, round(peak, 4)))

    if checked == 0:
        pytest.skip("no readable baseline checkpoints on disk")

    assert not ambiguous, (
        f"{len(ambiguous)} run(s) sit in the band where the threshold decides "
        f"the outcome: {ambiguous}. The threshold has stopped being informative "
        "for these; read them by hand and report what they show. Do NOT nudge "
        "MIN_TRAIN_MACRO_F1 to make this pass."
    )


@pytest.mark.parametrize("threshold", [0.2, 0.3, 0.4, 0.5])
def test_verdicts_are_stable_across_plausible_thresholds(threshold, monkeypatch):
    monkeypatch.setattr(C, "MIN_TRAIN_MACRO_F1", threshold)
    observed = {0.1519: False, 0.5154: True, 0.6882: True, 0.9968: True}
    for peak, expected in observed.items():
        assert C.converged(_history([peak])) is expected


# --------------------------------------------------------------------------
# Unknown is not failed
# --------------------------------------------------------------------------


def test_missing_history_is_unknown_not_failed():
    assert C.converged(None) is None
    assert C.converged([]) is None
    assert C.peak_train_macro_f1(None) is None


def test_history_for_a_missing_checkpoint_is_none(tmp_path):
    assert C.history_for("no_such_run", root=tmp_path) is None


def test_history_for_an_unreadable_checkpoint_is_none(tmp_path):
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "checkpoint.pt").write_bytes(b"not a torch file")
    assert C.history_for("broken", root=tmp_path) is None


def test_history_for_reads_a_real_checkpoint(tmp_path):
    import torch

    (tmp_path / "run").mkdir()
    history = _history([0.1, 0.9])
    torch.save({"history": history}, tmp_path / "run" / "checkpoint.pt")
    assert C.history_for("run", root=tmp_path) == history
    assert C.converged(C.history_for("run", root=tmp_path)) is True


def test_unknown_runs_stay_in_the_converged_subset():
    """Excluding what cannot be assessed would let pruning a checkpoint
    (T-208 does that routinely) quietly change a published number."""
    records = [
        {"seed": 0, "macro_f1": 0.1, "converged": None},
        {"seed": 1, "macro_f1": 0.2, "converged": True},
        {"seed": 2, "macro_f1": 0.3, "converged": False},
    ]
    out = C.summarise_convergence(records)
    assert out["nonconverged_seeds"] == [2]
    assert out["unknown_seeds"] == [0]
    assert [r["seed"] for r in out["converged_records"]] == [0, 1]
    assert out["n_converged"] == 2


def test_records_without_the_key_are_all_kept():
    """Older report rows predate the field; they must not read as failures."""
    records = [{"seed": s, "macro_f1": 0.5} for s in (0, 1, 2)]
    out = C.summarise_convergence(records)
    assert out["n_converged"] == 3
    assert out["nonconverged_seeds"] == []


# --------------------------------------------------------------------------
# How T-206 reports it
# --------------------------------------------------------------------------


def _rec(seed, f1, converged, peak=None):
    return dict(
        config="task3_ben_indicbert.yaml",
        run_id=f"task3_ben_indicbert_seed{seed}",
        seed=seed,
        split="task3/B/ben/native",
        encoder="indicbert-v2",
        batch_size=16,
        grad_accum=1,
        effective_batch=16,
        macro_f1=f1,
        accuracy=0.21,
        converged=converged,
        peak_train_macro_f1=peak,
    )


def test_summary_carries_both_figures():
    from scripts import t206_baseline as B

    records = [
        _rec(0, 0.0635, False, 0.1519),
        _rec(1, 0.1576, True, 0.9968),
        _rec(2, 0.1595, True, 0.6882),
    ]
    s = B.summarise(records)[0]
    assert s["macro_f1_mean"] == pytest.approx(0.1269, abs=1e-4)
    assert s["macro_f1_mean_converged"] == pytest.approx(0.1586, abs=1e-4)
    assert s["nonconverged_seeds"] == [0]
    assert s["n"] == 3 and s["n_converged"] == 2


def test_report_shows_both_and_names_the_failed_run(tmp_path):
    """Option chosen at T-402: report both, alongside — the all-seeds mean
    stays the headline and the converged subset sits next to it."""
    from scripts import t206_baseline as B

    records = [
        _rec(0, 0.0635, False, 0.1519),
        _rec(1, 0.1576, True, 0.9968),
        _rec(2, 0.1595, True, 0.6882),
    ]
    B.write_report(B.summarise(records), records, tmp_path)
    md = (tmp_path / "baselines.md").read_text()

    assert "0.1269" in md, "the all-seeds mean must remain the headline"
    assert "0.1585" in md or "0.1586" in md, "the converged figure sits alongside"
    assert "task3_ben_indicbert_seed0" in md, "the failed run is named"
    assert "0.1519" in md, "its peak train F1 is shown, not just a verdict"
    assert "did not fit their training data" in md


def test_report_says_so_when_every_seed_converged(tmp_path):
    from scripts import t206_baseline as B

    records = [_rec(s, 0.82, True, 0.99) for s in (0, 1, 2)]
    B.write_report(B.summarise(records), records, tmp_path)
    md = (tmp_path / "baselines.md").read_text()
    assert "all seeds converged" in md
    assert "did not fit their training data" not in md
