"""Tests for Phase 7 (`src/findings.py`, CLAUDE7.md).

The failure modes worth guarding against are quiet ones: a label that goes
uncounted, a bootstrap that treats one item's three seeds as three draws, a
verdict rule that rounds "one encoder" up to "held", and an artefact-driven
difference reported without its caveat.
"""

import pandas as pd
import pytest

from src import findings as F


def make_rows(spec: dict[str, list[str]], *, condition="c1", target="ben", seeds=(0,)) -> pd.DataFrame:
    """`spec` maps item id -> its label on each seed, in `seeds` order."""
    out = []
    for item, labels in spec.items():
        for seed, label in zip(seeds, labels):
            out.append({"condition_id": condition, "item_id": item, "seed": seed, "assigned_label": label,
                        "target": target, "untrained": False})
    return pd.DataFrame(out)


def shares(table: pd.DataFrame, level: str) -> dict[str, float]:
    t = table[table["level"] == level]
    return dict(zip(t["measure"], t["share"]))


# --------------------------------------------------------------------------
# share_table
# --------------------------------------------------------------------------


def test_every_level_sums_to_one_and_level_one_is_summed_from_level_two():
    rows = make_rows({"a": ["morphological_masking"], "b": ["class_collapse"], "c": ["seed_unstable"],
                      "d": ["unattributed"], "e": ["provenance_specific"]})
    table = F.share_table(rows, n_boot=50)
    l1, l2 = shares(table, "L1"), shares(table, "L2")
    assert sum(l1.values()) == pytest.approx(1.0)
    assert sum(l2.values()) == pytest.approx(1.0)
    assert l1["model"] == pytest.approx(l2["class_collapse"] + l2["seed_unstable"])
    assert l1["linguistic"] == pytest.approx(0.2)
    assert l1["unattributed"] == l2["unattributed"]
    assert F.check_shares_sum(table.assign(k=1), ["k"]) == []


def test_a_missing_share_is_caught():
    table = F.share_table(make_rows({"a": ["class_collapse"], "b": ["unattributed"]}), n_boot=20)
    table.loc[table["measure"] == "class_collapse", "share"] = 0.0
    assert F.check_shares_sum(table.assign(k=1), ["k"])


def test_the_bootstrap_unit_is_the_item_not_the_row():
    # one item failing on three seeds is one cluster, not three draws
    rows = make_rows({"a": ["class_collapse"] * 3, "b": ["unattributed"] * 3}, seeds=(0, 1, 2))
    table = F.share_table(rows, n_boot=20)
    assert set(table["n_units"]) == {2}
    assert set(table["n_failures"]) == {6}


def test_the_same_item_in_two_conditions_is_two_clusters():
    rows = pd.concat([make_rows({"a": ["class_collapse"]}, condition="c1"),
                      make_rows({"a": ["class_collapse"]}, condition="c2")])
    assert set(F.share_table(rows, n_boot=20)["n_units"]) == {2}


def test_the_interval_contains_the_point_and_is_deterministic():
    rows = make_rows({f"i{k}": ["class_collapse" if k % 3 else "unattributed"] for k in range(60)})
    first, second = F.share_table(rows, n_boot=200, seed=7), F.share_table(rows, n_boot=200, seed=7)
    pd.testing.assert_frame_equal(first, second)
    assert ((first["ci_low"] <= first["share"] + 1e-12) & (first["share"] <= first["ci_high"] + 1e-12)).all()


def test_per_seed_mean_and_spread():
    rows = make_rows({"a": ["class_collapse", "unattributed"], "b": ["class_collapse", "class_collapse"]},
                     seeds=(0, 1))
    row = F.share_table(rows, n_boot=20).set_index(["level", "measure"]).loc[("L2", "class_collapse")]
    assert row["seed_mean"] == pytest.approx(0.75)  # seed 0: 1.0, seed 1: 0.5
    assert row["seed_std"] == pytest.approx(0.3535534, rel=1e-6)
    assert row["n_seeds"] == 2


def test_zero_failures_is_refused_rather_than_divided_by():
    with pytest.raises(ValueError):
        F.share_table(make_rows({}).reindex(columns=["condition_id", "item_id", "seed", "assigned_label"]))


# --------------------------------------------------------------------------
# The rule (hard rule 5)
# --------------------------------------------------------------------------


def test_touching_intervals_overlap():
    assert not F.separated(0.1, 0.2, 0.2, 0.3)
    assert F.separated(0.1, 0.2, 0.21, 0.3)
    assert F.separated(0.5, 0.6, 0.1, 0.2)


@pytest.mark.parametrize(
    "results, expected",
    [
        ([(True, 0.1), (True, 0.2)], "held"),
        ([(True, -0.1), (True, -0.3)], "held"),
        ([(True, 0.1), (False, 0.2)], "one encoder only"),
        ([(True, 0.1), (True, -0.2)], "encoders disagree"),
        ([(False, 0.1), (False, -0.2)], "within noise"),
    ],
)
def test_verdict(results, expected):
    assert F.verdict(results, "encoder") == expected


def test_compare_separates_a_large_difference_and_not_an_identical_one():
    a = {f"a{k}": ["class_collapse"] for k in range(80)}
    b = {f"b{k}": ["unattributed"] for k in range(80)}
    rows = pd.concat([make_rows(a, condition="x", target="tel"), make_rows(b, condition="y", target="ben")])
    rows["encoder"] = "e1"
    out = F.compare(rows, a={"condition_id": ["x"]}, b={"condition_id": ["y"]}, replicate="encoder", n_boot=200)
    out = out.set_index(["level", "measure"])
    assert out.loc[("L2", "class_collapse"), "separated"]
    assert out.loc[("L1", "data"), "separated"] == False  # both 0%
    assert out.loc[("L2", "class_collapse"), "a_targets"] == "tel"
    assert ("L2", "unattributed") not in out.index  # it is L1 unattributed again


def test_compare_refuses_an_empty_group():
    rows = make_rows({"a": ["class_collapse"]}).assign(encoder="e1")
    with pytest.raises(ValueError):
        F.compare(rows, a={"condition_id": ["c1"]}, b={"condition_id": ["nope"]}, replicate="encoder", n_boot=10)


# --------------------------------------------------------------------------
# Caveats and runs that never trained
# --------------------------------------------------------------------------


def test_the_hindi_blind_spot_is_flagged_only_where_it_applies():
    assert "hindi_morphology" in F.caveats_for("T-702", "Task 2: x", "morphological_masking", "tel", "hin")
    assert "hindi_morphology" in F.caveats_for("T-702", "Task 2: x", "linguistic", "hin,tel", "ben")
    assert "hindi_morphology" not in F.caveats_for("T-702", "Task 2: x", "morphological_masking", "tel", "ben")
    assert "hindi_morphology" not in F.caveats_for("T-702", "Task 2: x", "class_collapse", "tel", "hin")


def test_native_drift_and_digit_caveats_belong_to_the_provenance_question():
    assert "native_drift" in F.caveats_for("T-705", "Task 2: x", "translation_drift", "tel", "tel")
    assert "native_drift" not in F.caveats_for("T-702", "Task 2: x", "translation_drift", "tel", "ben")
    assert "bengali_digits" in F.caveats_for("T-705", "Task 2: x", "class_collapse", "ben", "ben")
    assert "task3_ceiling" in F.caveats_for("T-703", "Task 3: x", "class_collapse", "ben", "tel")


def test_untrained_runs_are_matched_by_task_source_encoder_and_seed():
    rows = pd.DataFrame({
        "task": [3, 3, 3, 2],
        "source": ["tel", "tel", "ben", "tel"],
        "encoder": ["indicbert-v2", "indicbert-v2", "indicbert-v2", "indicbert-v2"],
        "seed": [0, 1, 0, 0],
    })
    mask = F.untrained_mask(rows, {"task3_tel_indicbert": {0, 2}})
    assert mask.tolist() == [True, False, False, False]


# --------------------------------------------------------------------------
# Reconciliation (hard rule 2)
# --------------------------------------------------------------------------


def test_reconcile_reports_a_failure_without_a_label(tmp_path, monkeypatch):
    path = tmp_path / "c1.parquet"
    pd.DataFrame({"item_id": ["a", "b"], "seed": [0, 0]}).to_parquet(path)
    monkeypatch.setattr(F, "_conditions", lambda task: [{"name": "c1"}])
    monkeypatch.setattr(F, "failure_path", lambda task, name, encoder=None: path)
    labelled = make_rows({"a": ["class_collapse"]})
    assert F.reconcile(2, "indicbert-v2", labelled) == ["c1: 1 failures without a label, 0 labels without a failure"]
    assert F.reconcile(2, "indicbert-v2", make_rows({"a": ["x"], "b": ["y"]})) == []


# --------------------------------------------------------------------------
# The committed report (T-707)
# --------------------------------------------------------------------------


def test_the_committed_findings_report_matches_its_artefacts():
    from scripts import t707_findings as T

    if not T.REPORT.exists() or not (T.PHASE7 / "robustness.parquet").exists():
        pytest.skip("Phase 7 artefacts not built")
    assert T.build(T.load()) == T.REPORT.read_text(), "run python -m scripts.t707_findings"
