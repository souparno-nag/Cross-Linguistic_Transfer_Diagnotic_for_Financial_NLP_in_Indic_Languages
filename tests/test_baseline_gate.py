"""Tests for T-207's baseline gate."""

import pandas as pd
import pytest

from src import baseline_gate as G


def _published(**overrides):
    base = {
        "source": "test",
        "tolerance_f1": 0.02,
        "baselines": {
            "cfg_a": {"task": 2, "published_macro_f1": 0.86, "note": "", "diagnosis": ""},
            "cfg_b": {"task": 3, "published_macro_f1": 0.05, "note": "", "diagnosis": ""},
        },
    }
    for k, v in overrides.items():
        base["baselines"][k].update(v)
    return base


def _ours(a_mean=None, b_mean=None):
    out = {}
    if a_mean is not None:
        out["cfg_a"] = {"n": 3, "seeds": [0, 1, 2], "split": "s",
                        "macro_f1_mean": a_mean, "macro_f1_std": 0.01, "accuracy_mean": a_mean}
    if b_mean is not None:
        out["cfg_b"] = {"n": 3, "seeds": [0, 1, 2], "split": "s",
                        "macro_f1_mean": b_mean, "macro_f1_std": 0.02, "accuracy_mean": 0.2}
    return out


def test_within_tolerance_passes():
    rows = {r["config"]: r for r in G.evaluate(_published(), _ours(a_mean=0.85, b_mean=0.15))}
    assert rows["cfg_a"]["passed"] and "within" in rows["cfg_a"]["status"]
    assert rows["cfg_b"]["passed"] and "above" in rows["cfg_b"]["status"]
    assert G.gate_passes(list(rows.values()))


def test_below_tolerance_without_diagnosis_fails():
    rows = G.evaluate(_published(), _ours(a_mean=0.80, b_mean=0.06))
    a = next(r for r in rows if r["config"] == "cfg_a")
    assert not a["passed"]
    assert "NO diagnosis" in a["status"]
    assert not G.gate_passes(rows)


def test_below_tolerance_with_diagnosis_passes():
    pub = _published(cfg_a={"diagnosis": "smaller training split than the paper; see notes"})
    rows = G.evaluate(pub, _ours(a_mean=0.80, b_mean=0.06))
    a = next(r for r in rows if r["config"] == "cfg_a")
    assert a["passed"]
    assert a["status"] == "below — diagnosed"
    assert G.gate_passes(rows)


def test_missing_run_is_a_failure():
    rows = G.evaluate(_published(), _ours(a_mean=0.86))  # cfg_b never run
    b = next(r for r in rows if r["config"] == "cfg_b")
    assert not b["passed"]
    assert "missing" in b["status"]


def test_our_results_aggregates_per_config(tmp_path):
    path = tmp_path / "baselines.parquet"
    pd.DataFrame(
        [
            {"config": "cfg_a", "run_id": "a0", "seed": 0, "split": "task2/H/hin/native", "encoder": "e", "macro_f1": 0.80, "accuracy": 0.81},
            {"config": "cfg_a", "run_id": "a1", "seed": 1, "split": "task2/H/hin/native", "encoder": "e", "macro_f1": 0.84, "accuracy": 0.85},
        ]
    ).to_parquet(path, index=False)

    res = G.our_results(path)
    assert res["cfg_a"]["n"] == 2
    assert res["cfg_a"]["macro_f1_mean"] == pytest.approx(0.82)
    assert res["cfg_a"]["seeds"] == [0, 1]


def test_shipped_published_baselines_file_parses():
    pub = G.load_published()
    assert "task2_hin_indicbert" in pub["baselines"]
    assert "task3_hin_indicbert" in pub["baselines"]
    assert pub["tolerance_f1"] == 0.02
