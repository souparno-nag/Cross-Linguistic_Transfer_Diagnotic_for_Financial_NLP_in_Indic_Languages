"""T-408's source-selection table (CLAUDE6.md).

The failure this file guards against is a table that quietly answers a
different question from the one it prints at the top. Three ways that can
happen, each with tests below:

* grouping on the wrong axis, because a prediction row's `block_id` names the
  block being *evaluated* rather than the one trained on;
* ranking sources by their raw target score, which rewards a source for
  starting high rather than for transferring well;
* letting a run that never trained into the mean, where its near-zero gap
  makes it the apparent best source.
"""

from __future__ import annotations

import pytest

from src import source_comparison as SC


def _cell(encoder, source, target, gap, std=0.02, status="ok", same_family=None):
    from src.conditions import FAMILY

    return {
        "task": 2,
        "encoder": encoder,
        "source": source,
        "target": target,
        "quadrant": f"{FAMILY[source]}->{FAMILY[target]}",
        "same_family": (
            FAMILY[source] == FAMILY[target] if same_family is None else same_family
        ),
        "n_excluded": 0,
        "excluded_seeds": [],
        "status": status,
        "n_seeds": 3,
        "n_items": 100,
        "source_mean": 0.95,
        "target_mean": 0.95 - gap,
        "gap_mean": gap,
        "gap_std": std,
        "within_seed_noise": False,
    }


# --------------------------------------------------------------------------
# The source axis
# --------------------------------------------------------------------------


def test_source_comes_from_the_condition_not_the_eval_block():
    """`transfer_ben_to_hin` evaluates on block H, so a prediction row's
    `block_id` reads 'H'. Grouping a source table on that would file Bengali's
    result under Hindi — the exact confusion CLAUDE6.md's Outputs note warns
    about."""
    condition = {
        "name": "transfer_ben_to_hin",
        "train": "task_2/B/ben",
        "eval": "task_2/H/hin",
    }
    assert SC.source_of(condition) == "ben"
    assert SC.target_of(condition) == "hin"


def test_family_membership_matches_the_conditions_table():
    from src.conditions import FAMILY

    assert SC.same_family("hin", "ben") is True      # both Indo-Aryan
    assert SC.same_family("tel", "mal") is True      # both Dravidian
    assert SC.same_family("hin", "tel") is False
    assert FAMILY["mal"] == "Dravidian"


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------


def test_sources_rank_by_gap_lower_is_better():
    rows = [
        _cell("e", "ben", "hin", 0.10), _cell("e", "ben", "tel", 0.20),
        _cell("e", "hin", "ben", 0.40), _cell("e", "hin", "tel", 0.50),
    ]
    ranked = SC.rank_sources(rows)
    assert [r["source"] for r in ranked] == ["ben", "hin"]
    assert ranked[0]["mean_gap"] == pytest.approx(0.15)
    assert ranked[0]["best_target"] == "hin" and ranked[0]["worst_target"] == "tel"


def test_ranking_uses_the_gap_not_the_target_score():
    """A weak source evaluated on an easy target can post a higher target score
    than a strong source on a hard one. Ranking on the score would invert the
    answer; the gap holds the source's own ceiling fixed."""
    strong = _cell("e", "ben", "hin", 0.10)
    weak = _cell("e", "hin", "ben", 0.40)
    strong["target_mean"], weak["target_mean"] = 0.50, 0.80  # weak scores higher
    ranked = SC.rank_sources([strong, weak])
    assert ranked[0]["source"] == "ben", "ranked on target score instead of gap"


def test_blocked_cells_are_excluded_from_ranking():
    rows = [_cell("e", "ben", "hin", 0.1), _cell("e", "hin", "ben", 0.0, status="blocked")]
    ranked = SC.rank_sources(rows)
    assert [r["source"] for r in ranked] == ["ben"]


# --------------------------------------------------------------------------
# Typological proximity — the question, asked honestly
# --------------------------------------------------------------------------


def test_proximity_predicts_when_the_same_family_source_wins():
    rows = [_cell("e", "ben", "hin", 0.10), _cell("e", "tel", "hin", 0.30)]
    out = SC.best_source_per_target(rows)[0]
    assert out["best_source"] == "ben"
    assert out["proximity_predicts"] is True


def test_proximity_fails_when_a_cross_family_source_wins():
    """Malayalam is Dravidian, so proximity says Telugu. If Bengali wins, the
    prediction failed and the table must say so."""
    rows = [
        _cell("e", "tel", "mal", 0.42),
        _cell("e", "ben", "mal", 0.37),
        _cell("e", "hin", "mal", 0.39),
    ]
    out = SC.best_source_per_target(rows)[0]
    assert out["best_source"] == "ben"
    assert out["proximity_predicts"] is False


def test_proximity_is_not_asked_when_no_same_family_source_exists():
    """Telugu's only possible sources are Hindi and Bengali, both Indo-Aryan —
    Malayalam has no native split and is never a source. Reporting 'no' there
    would be answering a question the data cannot pose."""
    rows = [_cell("e", "ben", "tel", 0.40), _cell("e", "hin", "tel", 0.45)]
    out = SC.best_source_per_target(rows)[0]
    assert out["same_family_available"] is False
    assert out["proximity_predicts"] is None


def test_encoders_are_ranked_separately():
    rows = [
        _cell("a", "ben", "hin", 0.10), _cell("a", "tel", "hin", 0.30),
        _cell("b", "ben", "hin", 0.30), _cell("b", "tel", "hin", 0.10),
    ]
    out = {r["encoder"]: r for r in SC.best_source_per_target(rows)}
    assert out["a"]["best_source"] == "ben"
    assert out["b"]["best_source"] == "tel"


# --------------------------------------------------------------------------
# Asymmetry
# --------------------------------------------------------------------------


def test_asymmetry_pairs_both_directions_and_names_the_harder_one():
    rows = [_cell("e", "hin", "ben", 0.45, std=0.05),
            _cell("e", "ben", "hin", 0.26, std=0.04)]
    out = SC.asymmetries(rows)
    assert len(out) == 1
    assert out[0]["difference"] == pytest.approx(0.19)
    assert out[0]["harder_direction"] == "hin->ben"
    assert out[0]["beats_noise"] is True


def test_a_small_asymmetry_does_not_beat_seed_noise():
    """Hard rule 5 applied to the comparison: a difference inside the two
    cells' combined spread is not a finding."""
    rows = [_cell("e", "hin", "ben", 0.40, std=0.10),
            _cell("e", "ben", "hin", 0.38, std=0.10)]
    assert SC.asymmetries(rows)[0]["beats_noise"] is False


def test_malayalam_never_appears_as_a_pair():
    """It has no native split, so it is never a source and cannot have a
    reverse direction."""
    rows = [_cell("e", "hin", "mal", 0.39), _cell("e", "ben", "mal", 0.37)]
    assert SC.asymmetries(rows) == []


def test_each_pair_is_reported_once():
    rows = [_cell("e", "hin", "ben", 0.45), _cell("e", "ben", "hin", 0.26),
            _cell("e", "hin", "tel", 0.45), _cell("e", "tel", "hin", 0.36)]
    pairs = [r["pair"] for r in SC.asymmetries(rows)]
    assert len(pairs) == len(set(pairs)) == 2


# --------------------------------------------------------------------------
# Target-family effect
# --------------------------------------------------------------------------


def test_target_family_effect_pools_over_source_family():
    """The question is whether the *target* is what costs, so Dravidian-source
    and Indo-Aryan-source cells must land in the same bucket when their target
    family matches. Splitting by both would just restate the quadrants."""
    rows = [
        _cell("e", "hin", "tel", 0.45),   # IA -> Dr
        _cell("e", "tel", "mal", 0.42),   # Dr -> Dr
        _cell("e", "ben", "hin", 0.26),   # IA -> IA
        _cell("e", "tel", "hin", 0.36),   # Dr -> IA
    ]
    out = {r["target_family"]: r for r in SC.target_family_effect(rows)}
    assert out["Dravidian"]["n_cells"] == 2
    assert out["Indo-Aryan"]["n_cells"] == 2
    assert out["Dravidian"]["mean_gap"] == pytest.approx(0.435)
    assert out["Indo-Aryan"]["mean_gap"] == pytest.approx(0.31)


# --------------------------------------------------------------------------
# Non-converged seeds
# --------------------------------------------------------------------------


def test_nonconverged_seeds_are_dropped_from_the_cell():
    """The one sanctioned exclusion. A model that never trained has a near-zero
    gap and would rank as the best source — it inverted task 3's Bengali
    comparison at T-404 before this existed."""
    rows = SC.cells(
        2, encoders=("indicbert-v2",),
        nonconverged={"task2_ben_indicbert": {0, 1}},
        n_boot=10,  # the numbers are not under test here, only the bookkeeping
    )
    ben = [r for r in rows if r["source"] == "ben"]
    assert ben, "expected Bengali-source cells"
    for r in ben:
        assert r["excluded_seeds"] == [0, 1]
        assert r["n_excluded"] == 2
        if r["status"] == "ok":
            assert r["n_seeds"] == 1


def test_a_source_with_every_seed_excluded_is_blocked_not_silently_dropped():
    rows = SC.cells(
        2, encoders=("indicbert-v2",),
        nonconverged={"task2_ben_indicbert": {0, 1, 2}},
        n_boot=10,
    )
    ben = [r for r in rows if r["source"] == "ben"]
    assert ben and all(r["status"] == "blocked" for r in ben)
    assert all(r["n_excluded"] == 3 for r in ben)


def test_exclusions_do_not_leak_across_encoders_or_tasks():
    """The key is the full run-id prefix, so excluding an IndicBERT seed must
    not remove the same seed from mBERT's cells."""
    rows = SC.cells(2, nonconverged={"task2_ben_indicbert": {0}}, n_boot=10)
    for r in rows:
        expected = 1 if (r["encoder"] == "indicbert-v2" and r["source"] == "ben") else 0
        assert r["n_excluded"] == expected


# --------------------------------------------------------------------------
# The lost encoder stays visible
# --------------------------------------------------------------------------


def test_xlmr_is_carried_as_a_stated_exclusion():
    """A dropped row hides the loss; CLAUDE5.md T-504 made the same choice."""
    assert "xlm-r-base" in SC.EXCLUDED_ENCODERS
    assert "xlm-r-base" not in SC.ENCODERS
    assert "VRAM" in SC.EXCLUDED_ENCODERS["xlm-r-base"]


def test_report_reads_convergence_from_the_baseline_artefact(tmp_path):
    """T-408 must agree with reports/baselines.md about which seeds trained.
    Re-deriving it from checkpoints let the two disagree — and cost more time
    than every bootstrap in the report combined."""
    import pandas as pd

    from scripts import t408_source_comparison as T

    path = tmp_path / "baselines.parquet"
    pd.DataFrame(
        [
            {"run_id": "task3_ben_indicbert_seed0", "seed": 0, "converged": False},
            {"run_id": "task3_ben_indicbert_seed1", "seed": 1, "converged": True},
            {"run_id": "task2_hin_mbert_seed2", "seed": 2, "converged": True},
        ]
    ).to_parquet(path, index=False)

    assert T.find_nonconverged(path) == {"task3_ben_indicbert": {0}}
