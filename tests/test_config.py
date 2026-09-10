"""Tests for T-204's config system.

Fast block: the YAML loader coerces types, rejects unknown keys, and the run
hash covers everything that moves a metric — including the data split, not just
the hyperparameters. Slow block is T-204's done criterion: two runs from one
config log identical metrics.
"""

import pytest

from src import config as C
from src import experiments as X

TASK2_YAML = "configs/train/task2_hin_indicbert.yaml"
TASK3_YAML = "configs/train/task3_hin_indicbert.yaml"


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def test_shipped_configs_load():
    r2 = C.load_run_config(TASK2_YAML)
    r3 = C.load_run_config(TASK3_YAML)
    assert r2.split_id() == "task2/H/hin/native"
    assert r3.split_id() == "task3/H/hin/native"
    assert r2.train_config().num_labels == 2
    assert r3.train_config().num_labels == 10


def test_yaml_scientific_notation_becomes_a_float(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("encoder: mbert-base\ntask: 2\nlang: hin\nlr: 3e-5\nepochs: 4\n")
    cfg = C.load_run_config(p)
    assert isinstance(cfg.lr, float)
    assert cfg.lr == pytest.approx(3e-5)
    assert isinstance(cfg.epochs, int)


def test_overrides_win_over_the_file():
    a = C.load_run_config(TASK2_YAML)
    b = C.load_run_config(TASK2_YAML, seed=2, epochs=1)
    assert (b.seed, b.epochs) == (2, 1)
    assert a.hash() != b.hash()


def test_unknown_key_is_rejected(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("encoder: mbert-base\ntask: 2\nlang: hin\nlearning_rate: 3e-5\n")
    with pytest.raises(ValueError, match="unknown keys"):
        C.load_run_config(p)


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(task=1), "no labels|numeral"),
        (dict(lang="mal"), "no block|no native"),
        (dict(dev_fraction=0.0), "dev_fraction"),
        (dict(dev_fraction=0.6, test_fraction=0.5), "no training data"),
        (dict(origin="bogus"), "native|mt"),
    ],
)
def test_invalid_configs_raise(kwargs, match):
    base = dict(encoder="mbert-base", task=2, lang="hin")
    with pytest.raises(ValueError, match=match):
        C.RunConfig(**{**base, **kwargs})


def test_malayalam_is_fine_with_an_explicit_block():
    cfg = C.RunConfig(encoder="mbert-base", task=2, lang="mal", origin="mt", block="H")
    assert cfg.split_id() == "task2/H/mal/mt"


# --------------------------------------------------------------------------
# Hashing
# --------------------------------------------------------------------------


def test_hash_is_stable_and_resolves_equivalent_specs():
    a = C.RunConfig(encoder="mbert-base", task=2, lang="hin")
    b = C.RunConfig(encoder="mbert-base", task=2, lang="hin", block="H")
    assert a.hash() == a.hash()
    assert a.hash() == b.hash()  # block=None and block="H" are the same split


@pytest.mark.parametrize(
    "field,value",
    [
        ("encoder", "xlm-r-base"),
        ("task", 3),
        ("lr", 5e-5),
        ("seed", 1),
        ("max_len", 64),
        ("dev_fraction", 0.2),
        ("test_fraction", 0.2),
        ("batch_size", 8),
    ],
)
def test_every_metric_changing_field_moves_the_run_hash(field, value):
    base = C.RunConfig(encoder="mbert-base", task=2, lang="hin")
    changed = C.RunConfig(
        **{**{f: getattr(base, f) for f in ("encoder", "task", "lang")}, field: value}
    )
    assert changed.hash() != base.hash()


# --------------------------------------------------------------------------
# experiments.csv
# --------------------------------------------------------------------------


def test_log_run_writes_header_once_then_appends(tmp_path):
    path = tmp_path / "experiments.csv"
    cfg = C.load_run_config(TASK2_YAML)
    X.log_run(cfg, "r1", {"accuracy": 0.9, "macro_f1": 0.88}, path=path)
    X.log_run(cfg, "r2", {"accuracy": 0.91, "macro_f1": 0.89}, path=path)

    rows = X.read_results(path)
    assert [r["run_id"] for r in rows] == ["r1", "r2"]
    assert rows[0]["config_hash"] == cfg.hash()
    assert rows[0]["split"] == "task2/H/hin/native"
    assert path.read_text().count("run_id,encoder,split") == 1


def test_logged_metrics_are_rounded_and_dated(tmp_path):
    path = tmp_path / "e.csv"
    cfg = C.load_run_config(TASK2_YAML)
    row = X.log_run(cfg, "r1", {"accuracy": 0.123456789, "macro_f1": 0.5}, path=path)
    assert row["accuracy"] == 0.123457
    assert row["date"].endswith("+00:00")


# --------------------------------------------------------------------------
# slow — T-204 done criterion
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_two_runs_from_one_config_produce_identical_metrics():
    run = C.RunConfig(
        encoder="mbert-base",
        task=3,
        lang="hin",
        max_len=64,
        batch_size=32,
        epochs=2,
        dev_fraction=0.2,
        test_fraction=0.2,
        seed=0,
    )
    a = C.run_training(run, device="cpu")
    b = C.run_training(run, device="cpu")
    assert a["config_hash"] == b["config_hash"]
    assert a["accuracy"] == b["accuracy"]
    assert a["macro_f1"] == b["macro_f1"]
    assert a["history"] == b["history"]
