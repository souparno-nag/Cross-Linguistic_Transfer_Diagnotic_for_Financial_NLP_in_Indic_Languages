"""Stage 3 — data and model-behaviour modules (src.behaviour, scripts.t609_behaviour).

Each module gets >= 5 positive and >= 5 negative fixtures (CLAUDE4.md T-607),
built as tiny synthetic prediction logs: one (item, seed) per case.
"""

from __future__ import annotations

import pandas as pd
import pytest

from scripts import t609_behaviour as T609
from src import behaviour as B
from src import diagnostics as D


def _log(rows):
    """rows: (item_id, seed, gold, pred)"""
    return pd.DataFrame(rows, columns=["item_id", "seed", "gold", "pred"])


# --------------------------------------------------------------------------
# provenance_specific — right in the twin condition
# --------------------------------------------------------------------------

def test_provenance_fires_when_the_twin_got_it_right_and_not_otherwise():
    keys = [(f"i{n}", 0) for n in range(10)]
    twin = _log([(f"i{n}", 0, 1, 1 if n < 5 else 0) for n in range(10)])
    assert B.provenance_status(keys, twin) == ["fired"] * 5 + ["not_fired"] * 5


def test_provenance_matches_on_seed_too():
    twin = _log([("i0", 1, 1, 1), ("i0", 0, 1, 0)])
    assert B.provenance_status([("i0", 0)], twin) == ["not_fired"]


def test_provenance_not_applicable_without_a_twin_or_the_item():
    assert B.provenance_status([("i0", 0)], None) == ["not_applicable"]
    assert B.provenance_status([("missing", 0)], _log([("i0", 0, 1, 1)])) == ["not_applicable"]


@pytest.mark.parametrize("cond,twin", [
    ("transfer_hin_to_ben", "transfer_hin_to_ben_mt"),
    ("transfer_hin_to_ben_mt", "transfer_hin_to_ben"),
    ("transfer_tel_to_mal", "transfer_tel_to_mal_mt"),
])
def test_twin_condition(cond, twin):
    assert B.twin_condition(cond) == twin


# --------------------------------------------------------------------------
# cross_lingual_hard — fails in all 3 targets from the same source
# --------------------------------------------------------------------------

def _audits(fails):
    """fails: {condition: [(item, seed), ...]}"""
    lang = lambda c: c.split("_to_")[1][:3]  # noqa: E731
    return {c: pd.DataFrame({"item_id": [i for i, _ in rows], "seed": [s for _, s in rows],
                             "tgt_lang": lang(c)}) for c, rows in fails.items()}


def test_xling_fires_only_with_all_three_targets():
    every = [(f"i{n}", 0) for n in range(5)]
    two = [(f"j{n}", 0) for n in range(5)]
    audits = _audits({
        "transfer_hin_to_ben": every + two,
        "transfer_hin_to_mal": every + two,
        "transfer_hin_to_tel": every,
    })
    failed = B.failed_languages(audits)
    fam = B.source_family("transfer_hin_to_ben")
    assert B.xling_status(every + two, fam, failed) == ["fired"] * 5 + ["not_fired"] * 5


def test_xling_keeps_native_and_mt_families_apart():
    audits = _audits({"transfer_hin_to_ben": [("i", 0)], "transfer_hin_to_mal_mt": [("i", 0)],
                      "transfer_hin_to_tel": [("i", 0)]})
    failed = B.failed_languages(audits)
    assert B.xling_status([("i", 0)], B.source_family("transfer_hin_to_ben"), failed) == ["not_fired"]


def test_xling_needs_the_same_seed():
    audits = _audits({"transfer_hin_to_ben": [("i", 0)], "transfer_hin_to_mal": [("i", 1)],
                      "transfer_hin_to_tel": [("i", 0)]})
    failed = B.failed_languages(audits)
    assert B.xling_status([("i", 0)], B.source_family("transfer_hin_to_ben"), failed) == ["not_fired"]


# --------------------------------------------------------------------------
# class_collapse — the run predicts one class >= 75% and this failure is it
# --------------------------------------------------------------------------

def test_collapse_fires_on_the_dominant_class_of_a_collapsed_run():
    # seed 0: 8/10 predict 1 (collapsed); seed 1: 5/10 (not)
    rows = [(f"i{n}", 0, 0, 1 if n < 8 else 0) for n in range(10)]
    rows += [(f"i{n}", 1, 0, 1 if n < 5 else 0) for n in range(10)]
    log = _log(rows)
    keys = [(f"i{n}", 0) for n in range(5)] + [(f"i{n}", 1) for n in range(5)]
    preds = [1] * 10
    assert B.collapse_status(keys, preds, log) == ["fired"] * 5 + ["not_fired"] * 5


def test_collapse_ignores_a_failure_on_the_minority_class():
    log = _log([(f"i{n}", 0, 1, 1 if n < 8 else 0) for n in range(10)])
    assert B.collapse_status([("i9", 0)], [0], log) == ["not_fired"]


def test_collapse_cutoff_is_inclusive():
    log = _log([(f"i{n}", 0, 1, 0 if n < 3 else 1) for n in range(4)])  # exactly 75% predict 0
    assert B.collapse_status([("i0", 0)], [0], log) == ["fired"]


# --------------------------------------------------------------------------
# seed_unstable — right on another seed of the same condition
# --------------------------------------------------------------------------

def test_seed_fires_when_another_seed_got_it_right():
    tgt = []
    src = []
    for n in range(10):
        tgt += [(f"i{n}", 0, 1, 0), (f"i{n}", 1, 1, 1 if n < 5 else 0)]
        src += [(f"i{n}", 0, 1, 1), (f"i{n}", 1, 1, 1)]
    keys = [(f"i{n}", 0) for n in range(10)]
    assert B.seed_status(keys, _log(tgt), _log(src)) == ["fired"] * 5 + ["not_fired"] * 5


def test_seed_ignores_a_seed_whose_source_was_wrong():
    tgt = _log([("i", 0, 1, 0), ("i", 1, 1, 1)])
    src = _log([("i", 0, 1, 1), ("i", 1, 1, 0)])
    assert B.seed_status([("i", 0)], tgt, src) == ["not_fired"]


def test_seed_does_not_count_the_failing_seed_itself():
    tgt = _log([("i", 0, 1, 1)])  # pathological: the "failure" is right on its own seed
    src = _log([("i", 0, 1, 1)])
    assert B.seed_status([("i", 0)], tgt, src) == ["not_fired"]


# --------------------------------------------------------------------------
# annotate / resolve — precedence, gate terminal, grouping, idempotence
# --------------------------------------------------------------------------

def _cell_audit():
    return pd.DataFrame({
        "item_id": ["a", "b", "c"], "seed": [0, 0, 0], "tgt_lang": "ben",
        "gate_status": ["fired", "passed", "passed"],
        "modules_fired": [["gate"], ["morph"], []],
        "assigned_label": ["translation_drift", "morphological_masking", "unattributed"],
    })


def _annotated():
    audit = _cell_audit()
    tgt = _log([("a", 0, 0, 1), ("b", 0, 0, 1), ("c", 0, 0, 1), ("c", 1, 0, 0)])
    src = _log([("a", 0, 0, 0), ("b", 0, 0, 0), ("c", 0, 0, 0), ("c", 1, 0, 0)])
    twin = _log([("a", 0, 0, 0), ("b", 0, 0, 0), ("c", 0, 0, 1)])
    return B.annotate(audit, "transfer_hin_to_ben", target_log=tgt, source_log=src, twin_log=twin,
                      failed_langs={})


def test_gate_fired_rows_are_not_evaluated():
    a = _annotated()
    assert set(a.loc[0, B.STATUS_COLUMNS]) == {"not_evaluated"}


def test_linguistic_label_outranks_stage_3_and_both_are_in_the_trail():
    a = _annotated()
    audit, labels = T609.resolve(a, _cell_audit()[["item_id", "seed", "modules_fired", "assigned_label"]])
    assert audit.loc[1, "assigned_label"] == "morphological_masking"
    assert "provenance" in audit.loc[1, "modules_fired"]
    assert audit.loc[1, "label_group"] == "linguistic"
    assert audit.loc[0, "assigned_label"] == "translation_drift"


def test_unattributed_row_is_claimed_by_stage_3_in_precedence_order():
    a = _annotated()  # row c: twin wrong; seed 1 right; seed 0 predicts class 1 on every row
    audit, labels = T609.resolve(a, _cell_audit()[["item_id", "seed", "modules_fired", "assigned_label"]])
    assert audit.loc[2, "modules_fired"] == ["collapse", "seed"]
    assert audit.loc[2, "assigned_label"] == "class_collapse"
    assert audit.loc[2, "label_group"] == "model"
    assert labels.loc[2, "assigned_label"] == "class_collapse"


def test_resolve_is_idempotent():
    a = _annotated()
    once, _ = T609.resolve(a, _cell_audit())
    twice, _ = T609.resolve(once, _cell_audit())
    assert once["modules_fired"].tolist() == twice["modules_fired"].tolist()
    assert once["assigned_label"].tolist() == twice["assigned_label"].tolist()


def test_every_precedence_label_has_a_group():
    for _, label in D.PRECEDENCE:
        assert D.label_group(label) in ("linguistic", "data", "model")
    assert D.label_group("unattributed") == "unattributed"
