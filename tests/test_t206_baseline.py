"""Tests for T-206's orchestration helpers (the pure parts, no training)."""

import pytest

from scripts import t206_baseline as B


class _Path:
    def __init__(self, stem):
        self.stem = stem
        self.name = f"{stem}.yaml"


def test_plan_runs_is_the_config_x_seed_grid():
    plan = B.plan_runs([_Path("task2_hin"), _Path("task3_hin")], [0, 1, 2])
    assert len(plan) == 6
    assert {p["run_id"] for p in plan} == {
        f"{c}_seed{s}" for c in ("task2_hin", "task3_hin") for s in (0, 1, 2)
    }


def test_already_done_matches_run_id_and_hash(tmp_path):
    from src import experiments as X

    class Cfg:
        encoder = "indicbert-v2"
        seed = 0

        def split_id(self):
            return "task2/H/hin/native"

        def hash(self):
            return "abc123"

    path = tmp_path / "e.csv"
    X.log_run(Cfg(), "task2_hin_seed0", {"accuracy": 0.9, "macro_f1": 0.88}, path=path)

    assert B.already_done("task2_hin_seed0", "abc123", path) is not None
    assert B.already_done("task2_hin_seed0", "different", path) is None
    assert B.already_done("task2_hin_seed1", "abc123", path) is None


def test_summarise_reports_mean_and_sample_std_per_config():
    records = [
        dict(config="a.yaml", run_id="a0", seed=0, split="s", encoder="e", macro_f1=0.80, accuracy=0.90),
        dict(config="a.yaml", run_id="a1", seed=1, split="s", encoder="e", macro_f1=0.82, accuracy=0.92),
        dict(config="a.yaml", run_id="a2", seed=2, split="s", encoder="e", macro_f1=0.84, accuracy=0.94),
        dict(config="b.yaml", run_id="b0", seed=0, split="s", encoder="e", macro_f1=0.50, accuracy=0.60),
    ]
    summary = {s["config"]: s for s in B.summarise(records)}

    a = summary["a.yaml"]
    assert a["n"] == 3
    assert a["seeds"] == [0, 1, 2]
    assert a["macro_f1_mean"] == pytest.approx(0.82)
    assert a["macro_f1_std"] == pytest.approx(0.02)
    assert a["macro_f1_by_seed"] == {0: 0.80, 1: 0.82, 2: 0.84}

    b = summary["b.yaml"]
    assert b["n"] == 1
    assert b["macro_f1_std"] == 0.0  # one seed: std undefined -> 0


def test_write_report_emits_md_and_parquet(tmp_path):
    import pandas as pd

    records = [
        dict(config="a.yaml", run_id="a0", seed=0, split="task2/H/hin/native", encoder="indicbert-v2", macro_f1=0.8, accuracy=0.9),
        dict(config="a.yaml", run_id="a1", seed=1, split="task2/H/hin/native", encoder="indicbert-v2", macro_f1=0.82, accuracy=0.91),
    ]
    B.write_report(B.summarise(records), records, tmp_path)

    md = (tmp_path / "baselines.md").read_text()
    assert "macro-F1 (mean ± std)" in md
    assert "a.yaml" in md
    assert len(pd.read_parquet(tmp_path / "baselines.parquet")) == 2
