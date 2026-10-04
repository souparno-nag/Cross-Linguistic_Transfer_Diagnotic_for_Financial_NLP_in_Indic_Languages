"""The small helpers behind the Phase 6 GPU runner (scripts/run_phase6_gpu.sh)."""

from __future__ import annotations

import pandas as pd
import pytest

from scripts import t601_diagnostics as T601
from scripts import t601_report as REP
from scripts import t607_spotcheck as SPOT


def test_spotcheck_cell_keeps_the_full_text_and_survives_a_markdown_table():
    long = "ক" * 300 + " | tail\nsecond line"
    cell = SPOT._cell(long)
    assert cell.endswith("tail second line") and "…" not in cell
    assert "\n" not in cell and "\\|" in cell and "|" not in cell.replace("\\|", "")


def test_no_silent_cpu_fallback(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(SystemExit, match="refusing to fall back to CPU"):
        T601.resolve_device(None)
    assert T601.resolve_device("cpu") == "cpu"  # deliberate CPU is allowed


def _audit(rows):
    cols = {"condition_id": "c", "assigned_label": "unattributed", "frag_status": "not_fired",
            "morph_status": "not_fired", "saliency_status": "not_applicable"}
    return pd.DataFrame([{**cols, **r} for r in rows])


def test_condition_table_reports_unattributed_rate_and_every_status():
    audit = _audit([
        {"assigned_label": "morphological_masking", "morph_status": "fired"},
        {"saliency_status": "not_converged"},
        {}, {},
    ])
    row = REP.condition_table(audit).iloc[0]
    assert row["n"] == 4 and row["unattributed_%"] == 75.0
    assert row["morph:fired"] == 1 and row["saliency:not_converged"] == 1


def test_threshold_pairs_exclude_unconverged_and_split_either_side(monkeypatch):
    rows = [{"saliency_status": "not_fired", "saliency_divergence": d, "item_id": f"i{i}",
             "block_id": "B", "tgt_lang": "ben", "saliency_convergence_error": 0.01}
            for i, d in enumerate([0.1, 0.25, 0.35, 0.5])]
    rows.append({"saliency_status": "not_converged", "saliency_divergence": 0.3, "item_id": "bad",
                 "block_id": "B", "tgt_lang": "ben", "saliency_convergence_error": 0.9})
    monkeypatch.setattr(REP, "load_matrix", lambda task: {"conditions": [
        {"name": "c", "train": "task_2/H/hin"}]})
    monkeypatch.setattr(REP, "read_split", lambda *a, **k: pd.DataFrame({"item_id": [], "text": []}))
    out = REP.nearest_to_threshold(2, _audit(rows), 0.3)
    assert "bad" not in out["item_id"].tolist()
    assert sorted(out["divergence"]) == [0.1, 0.25, 0.35, 0.5]
