"""Tests for T-601 (pipeline scaffold), T-605 (multi-fire audit trail) and
the precedence resolution, per CLAUDE4.md.

`diagnose_row` is pure (no I/O, no torch) given a stub tokenizer and a real
`MorphAnalyzer` (which degrades to rung 2 without `indic-nlp-library`
installed, exactly like `test_morphology.py`), so the router's decision
logic is fully covered without a GPU or a checkpoint. `loaded=None` makes
the saliency module report `unavailable` -- CLAUDE4.md never requires every
module to run, only that the ones that couldn't are declared so, not
silently skipped.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src import diagnostics as D
from src import morphology as M


class _WhitespaceTokenizer:
    def tokenize(self, text: str) -> list[str]:
        return text.split()


# --------------------------------------------------------------------------
# precedence_pick
# --------------------------------------------------------------------------


def test_precedence_gate_beats_everything():
    assert D.precedence_pick(["gate", "frag", "morph", "saliency"]) == "translation_drift"


def test_precedence_frag_beats_morph_and_saliency():
    assert D.precedence_pick(["frag", "morph", "saliency"]) == "tokenizer_fragmentation"


def test_precedence_morph_beats_saliency():
    assert D.precedence_pick(["morph", "saliency"]) == "morphological_masking"


def test_precedence_saliency_alone():
    assert D.precedence_pick(["saliency"]) == "terminology_gap"


def test_precedence_nothing_fired_is_unattributed():
    assert D.precedence_pick([]) == "unattributed"


# --------------------------------------------------------------------------
# diagnose_row
# --------------------------------------------------------------------------


@pytest.fixture()
def tokenizer():
    return _WhitespaceTokenizer()


def test_gate_fired_skips_every_stage_two_module(tokenizer):
    morph_analyzer = M.MorphAnalyzer("tel")
    row = D.diagnose_row(
        labse_sim=0.5, tau=0.82, source_text="a b", target_text="a b c d e f",
        gold_label=0, tokenizer=tokenizer, morph_analyzer=morph_analyzer,
        concepts=[], lang_src="hin", lang_tgt="tel", loaded=None,
    )
    assert row["gate_status"] == "fired"
    assert row["assigned_label"] == "translation_drift"
    assert row["modules_fired"] == ["gate"]
    assert row["frag_status"] == "not_evaluated"
    assert row["morph_status"] == "not_evaluated"
    assert row["saliency_status"] == "not_evaluated"


def test_gate_passed_and_nothing_fires_is_unattributed(tokenizer):
    morph_analyzer = M.MorphAnalyzer("tel")
    row = D.diagnose_row(
        labse_sim=0.95, tau=0.82, source_text="a b", target_text="a b",
        gold_label=0, tokenizer=tokenizer, morph_analyzer=morph_analyzer,
        concepts=[{"id": "c", "tel": "నహి"}], lang_src="hin", lang_tgt="tel", loaded=None,
    )
    assert row["gate_status"] == "passed"
    assert row["frag_status"] == "not_fired"
    assert row["morph_status"] == "not_fired"
    assert row["saliency_status"] == "unavailable"
    assert row["modules_fired"] == []
    assert row["assigned_label"] == "unattributed"


def test_fragmentation_fires_and_wins(tokenizer):
    morph_analyzer = M.MorphAnalyzer("tel")
    row = D.diagnose_row(
        labse_sim=0.95, tau=0.82, source_text="a b", target_text="a b c d e f",
        gold_label=0, tokenizer=tokenizer, morph_analyzer=morph_analyzer,
        concepts=[], lang_src="hin", lang_tgt="tel", loaded=None,
    )
    assert row["r_frag"] == pytest.approx(3.0)
    assert row["frag_status"] == "fired"
    assert row["assigned_label"] == "tokenizer_fragmentation"
    assert "frag" in row["modules_fired"]


def test_morphology_fires_when_frag_does_not(tokenizer):
    morph_analyzer = M.MorphAnalyzer("tel")
    concepts = [{"id": "resource_test", "tel": "వనరు"}]
    row = D.diagnose_row(
        labse_sim=0.95, tau=0.82, source_text="a b", target_text="వనరుకు కొరత",
        gold_label=0, tokenizer=tokenizer, morph_analyzer=morph_analyzer,
        concepts=concepts, lang_src="hin", lang_tgt="tel", loaded=None,
    )
    assert row["frag_status"] == "not_fired"
    assert row["morph_status"] == "fired"
    assert row["assigned_label"] == "morphological_masking"
    assert row["modules_fired"] == ["morph"]


def test_multi_fire_records_every_module_but_precedence_picks_one(tokenizer):
    """Both fragmentation and morphology fire; the audit trail records both
    but `assigned_label` follows precedence (frag before morph)."""
    morph_analyzer = M.MorphAnalyzer("tel")
    concepts = [{"id": "resource_test", "tel": "వనరు"}]
    row = D.diagnose_row(
        labse_sim=0.95, tau=0.82, source_text="a", target_text="a b c d వనరుకు కొరత",
        gold_label=0, tokenizer=tokenizer, morph_analyzer=morph_analyzer,
        concepts=concepts, lang_src="hin", lang_tgt="tel", loaded=None,
    )
    assert row["frag_status"] == "fired"
    assert row["morph_status"] == "fired"
    assert set(row["modules_fired"]) == {"frag", "morph"}
    assert row["assigned_label"] == "tokenizer_fragmentation"


# --------------------------------------------------------------------------
# discover_conditions
# --------------------------------------------------------------------------


def test_discover_conditions_matches_failures_files_to_the_matrix(tmp_path, monkeypatch):
    task_dir = tmp_path / "task_9"
    task_dir.mkdir(parents=True)
    (task_dir / "transfer_hin_to_ben.parquet").write_bytes(b"")

    fake_matrix = {
        "conditions": [
            {"name": "transfer_hin_to_ben", "kind": "transfer", "train": "task_9/H/hin", "eval": "task_9/B/ben"},
        ]
    }
    monkeypatch.setattr(D, "load_matrix", lambda task: fake_matrix)

    found = D.discover_conditions(9, root=tmp_path)
    assert found == [{"condition_id": "transfer_hin_to_ben", "block_src": "H", "lang_src": "hin"}]


def test_discover_conditions_raises_on_unknown_condition(tmp_path, monkeypatch):
    task_dir = tmp_path / "task_9"
    task_dir.mkdir(parents=True)
    (task_dir / "mystery_condition.parquet").write_bytes(b"")
    monkeypatch.setattr(D, "load_matrix", lambda task: {"conditions": []})

    with pytest.raises(ValueError, match="mystery_condition"):
        D.discover_conditions(9, root=tmp_path)


# --------------------------------------------------------------------------
# load_esg_terms
# --------------------------------------------------------------------------


def test_load_esg_terms_reads_the_real_lexicon():
    concepts = D.load_esg_terms()
    assert len(concepts) >= 15
    for concept in concepts:
        assert "id" in concept and "hin" in concept and concept["hin"]


def test_load_esg_terms_from_a_custom_path(tmp_path):
    path = tmp_path / "esg_terms.json"
    path.write_text(json.dumps({"concepts": [{"id": "x", "hin": "y"}]}))
    assert D.load_esg_terms(path) == [{"id": "x", "hin": "y"}]


# --------------------------------------------------------------------------
# write_diagnostics / read_diagnostics round trip
# --------------------------------------------------------------------------


def test_write_and_read_diagnostics_round_trip(tmp_path):
    labels = pd.DataFrame([{
        "condition_id": "c", "encoder_id": "indicbert-v2", "item_id": "i1",
        "block_id": "H", "src_lang": "hin", "tgt_lang": "tel",
        "assigned_label": "unattributed", "modules_fired": [], "r_frag": None,
        "labse_sim": 0.9, "seed": 0,
    }])
    audit = pd.DataFrame([{
        "condition_id": "c", "encoder_id": "indicbert-v2", "item_id": "i1",
        "block_id": "H", "src_lang": "hin", "tgt_lang": "tel", "seed": 0,
        "gate_status": "passed", "labse_sim": 0.9,
        "frag_status": "not_fired", "r_frag": 1.0,
        "morph_status": "not_fired", "morph_evidence": None,
        "saliency_status": "unavailable", "saliency_divergence": None,
        "saliency_convergence_error": None,
        "assigned_label": "unattributed", "modules_fired": [],
    }])
    label_path = D.diagnostics_path(1, "c", root=tmp_path / "diag")
    audit_path_ = D.audit_path(1, "c", root=tmp_path / "diag" / "audit")
    label_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path_.parent.mkdir(parents=True, exist_ok=True)
    labels[D.LABEL_COLUMNS].to_parquet(label_path, index=False)
    audit[D.AUDIT_COLUMNS].to_parquet(audit_path_, index=False)

    back = D.read_diagnostics(1, "c", root=tmp_path / "diag")
    assert back["item_id"].tolist() == ["i1"]
    assert back["assigned_label"].tolist() == ["unattributed"]


# --------------------------------------------------------------------------
# encoder-namespaced paths (a second encoder must not overwrite the first)
# --------------------------------------------------------------------------


def test_default_encoder_keeps_the_original_path_and_others_get_a_level(tmp_path):
    assert D.path_encoder("indicbert-v2") is None
    assert D.path_encoder("mbert-base") == "mbert-base"
    a = D.diagnostics_path(2, "c", root=tmp_path)
    b = D.diagnostics_path(2, "c", root=tmp_path, encoder="mbert-base")
    assert a != b and b.parent.name == "mbert-base" and a.parent.name == "task_2"
    assert D.audit_path(2, "c", root=tmp_path, encoder="mbert-base").parent.name == "mbert-base"


def test_discover_conditions_reads_an_encoder_subdirectory(tmp_path, monkeypatch):
    sub = tmp_path / "task_9" / "mbert-base"
    sub.mkdir(parents=True)
    (sub / "transfer_hin_to_ben.parquet").write_bytes(b"")
    monkeypatch.setattr(D, "load_matrix", lambda task: {"conditions": [
        {"name": "transfer_hin_to_ben", "train": "task_9/H/hin"}]})
    assert D.discover_conditions(9, root=tmp_path) == []
    assert [c["condition_id"] for c in D.discover_conditions(9, root=tmp_path, encoder="mbert-base")] == [
        "transfer_hin_to_ben"]
