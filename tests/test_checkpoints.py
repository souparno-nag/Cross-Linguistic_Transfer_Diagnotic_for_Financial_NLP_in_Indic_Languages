"""Tests for T-208's checkpoint registry and retention."""

import pytest
import torch

from src import checkpoints as C


def _fake_checkpoint(root, run_id, config_hash, *, run_hash=None, best_f1=0.8, epoch=3):
    d = root / run_id
    d.mkdir(parents=True)
    blob = {
        "config_hash": config_hash,
        "epoch": epoch,
        "history": [{"epoch": i} for i in range(epoch + 1)],
        "best": {"macro_f1": best_f1, "epoch": 1, "state": {"w": torch.zeros(2)}},
    }
    if run_hash is not None:
        blob["run_hash"] = run_hash
    torch.save(blob, d / "checkpoint.pt")
    return d


@pytest.fixture
def registry(tmp_path, monkeypatch):
    root = tmp_path / "checkpoints"
    root.mkdir()
    monkeypatch.setattr(C, "CHECKPOINT_ROOT", root)
    return root


def test_retrieval_by_run_id(registry):
    _fake_checkpoint(registry, "task2_hin_indicbert_seed1", "hash_a")
    assert C.exists("task2_hin_indicbert_seed1")
    assert C.checkpoint_path("task2_hin_indicbert_seed1").exists()
    assert C.load("task2_hin_indicbert_seed1")["config_hash"] == "hash_a"
    with pytest.raises(FileNotFoundError):
        C.load("nope_seed0")


def test_metadata_and_best_state(registry):
    _fake_checkpoint(registry, "r_seed0", "h", best_f1=0.77, epoch=5)
    meta = C.metadata("r_seed0")
    assert meta["config_hash"] == "h"
    assert meta["last_epoch"] == 5
    assert meta["epochs_recorded"] == 6
    assert meta["best_dev_macro_f1"] == 0.77
    assert meta["has_best_weights"] is True
    assert "w" in C.best_state_dict("r_seed0")


def test_list_all_finds_every_checkpoint(registry):
    _fake_checkpoint(registry, "a_seed0", "h1")
    _fake_checkpoint(registry, "b_seed0", "h2")
    (registry / "empty_dir").mkdir()  # no checkpoint.pt -> ignored
    ids = {m["run_id"] for m in C.list_all()}
    assert ids == {"a_seed0", "b_seed0"}


def test_retention_keeps_reported_and_current_drops_the_rest(registry, tmp_path, monkeypatch):
    # reported: run_hash is logged in experiments.csv
    _fake_checkpoint(registry, "task2_hin_indicbert_seed0", "train_r", run_hash="reported_run")
    # current: run_hash matches the live YAML's RunConfig hash
    _fake_checkpoint(registry, "task2_hin_indicbert_seed1", "train_c", run_hash="current_run")
    # orphan: neither
    _fake_checkpoint(registry, "task2_hin_indicbert_seed2", "train_x", run_hash="stale_run")

    results = tmp_path / "experiments.csv"
    results.write_text(
        "run_id,encoder,split,seed,config_hash,accuracy,macro_f1,date\n"
        "task2_hin_indicbert_seed0,e,s,0,reported_run,0.8,0.8,2026-01-01T00:00:00+00:00\n"
    )
    monkeypatch.setattr(
        C, "_current_hashes",
        lambda rid: (("current_run", "train_c")
                     if rid == "task2_hin_indicbert_seed1" else (None, None)),
    )

    groups = C.classify(results)
    assert {m["run_id"] for m in groups["keep"]} == {
        "task2_hin_indicbert_seed0",
        "task2_hin_indicbert_seed1",
    }
    assert [m["run_id"] for m in groups["orphan"]] == ["task2_hin_indicbert_seed2"]

    assert C.prune(results, dry_run=True) == ["task2_hin_indicbert_seed2"]
    assert C.exists("task2_hin_indicbert_seed2")  # dry run kept it

    assert C.prune(results, dry_run=False) == ["task2_hin_indicbert_seed2"]
    assert not C.exists("task2_hin_indicbert_seed2")
    assert C.exists("task2_hin_indicbert_seed0")


def test_current_hashes_resolve_from_a_shipped_config():
    """A real run_id maps back to its YAML and hashes to the same values."""
    from src.config import load_run_config

    rid = "task3_hin_indicbert_seed2"
    run = load_run_config("configs/train/task3_hin_indicbert.yaml", seed=2)
    assert C._current_hashes(rid) == (run.hash(), run.train_config().hash())
    assert C._current_hashes("no_such_config_seed0") == (None, None)
    assert C._current_hashes("malformed-run-id") == (None, None)
