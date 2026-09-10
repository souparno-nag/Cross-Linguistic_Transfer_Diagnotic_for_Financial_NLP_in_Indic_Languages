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


def test_report_records_takes_current_hash_and_spans_all_configs(tmp_path):
    """A partial re-run must not drop untouched configs; stale rows are ignored."""
    from src import experiments as X
    from src.config import TRAIN_CONFIG_DIR, load_run_config

    configs = sorted(TRAIN_CONFIG_DIR.glob("*.yaml"))
    assert len(configs) >= 2
    path = tmp_path / "e.csv"

    for cfg in configs:
        run = load_run_config(cfg, seed=0)
        X.log_run(run, f"{cfg.stem}_seed0", {"accuracy": 0.5, "macro_f1": 0.4}, path=path)

    # a stale row for the last config: same run_id, obsolete hash, better score
    class Stale:
        encoder = "indicbert-v2"
        seed = 0

        def split_id(self):
            return "x"

        def hash(self):
            return "obsoletehash"

    X.log_run(Stale(), f"{configs[-1].stem}_seed0", {"accuracy": 0.99, "macro_f1": 0.99}, path=path)

    recs = B.report_records(configs, [0], path)
    assert len(recs) == len(configs)
    assert all(r["macro_f1"] == 0.4 for r in recs)  # never the stale 0.99


def test_stale_checkpoint_detection(tmp_path):
    import torch

    class Run:
        def __init__(self, rh, th):
            self._rh, self._th = rh, th

        def hash(self):
            return self._rh

        def train_config(self):
            return type("T", (), {"hash": lambda _s: self._th})()

    run = Run("run_abc", "train_abc")
    wd = tmp_path / "run"
    wd.mkdir()
    assert B.stale_checkpoint(wd, run) is False  # no checkpoint -> not stale

    # current-format checkpoint: matched on run_hash
    torch.save({"config_hash": "train_abc", "run_hash": "run_abc"}, wd / "checkpoint.pt")
    assert B.stale_checkpoint(wd, run) is False
    assert B.stale_checkpoint(wd, Run("run_def", "train_def")) is True

    # legacy checkpoint (no run_hash): fall back to train-config hash
    torch.save({"config_hash": "train_abc"}, wd / "checkpoint.pt")
    assert B.stale_checkpoint(wd, run) is False
    assert B.stale_checkpoint(wd, Run("x", "train_def")) is True

    (wd / "checkpoint.pt").write_bytes(b"not a torch file")
    assert B.stale_checkpoint(wd, run) is True  # unreadable -> redo


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
