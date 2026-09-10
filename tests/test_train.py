"""Tests for T-203's training loop.

The fast block checks the parts that must be right before a GPU is worth
booking: the config hash catches every metric-changing knob, AdamW groups
params the standard way, the seed is honoured. The ``slow`` block is the
acceptance criterion itself — the loop must memorise a tiny subset — plus
determinism and checkpoint-resume.
"""

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from src import train as T


# --------------------------------------------------------------------------
# Fast — no model download
# --------------------------------------------------------------------------


def base_config():
    return T.TrainConfig(encoder="mbert-base", num_labels=2)


def test_config_hash_is_stable():
    assert base_config().hash() == base_config().hash()


@pytest.mark.parametrize(
    "field,value",
    [
        ("encoder", "xlm-r-base"),
        ("num_labels", 10),
        ("lr", 1e-5),
        ("batch_size", 8),
        ("grad_accum", 4),
        ("max_len", 64),
        ("seed", 1),
        ("warmup_ratio", 0.2),
    ],
)
def test_every_metric_changing_knob_moves_the_hash(field, value):
    assert replace(base_config(), **{field: value}).hash() != base_config().hash()


def test_adamw_excludes_bias_and_layernorm_from_decay():
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.dense = nn.Linear(4, 4)
            self.LayerNorm = nn.LayerNorm(4)

    groups = T._param_groups(Tiny(), weight_decay=0.01)
    assert groups[0]["weight_decay"] == 0.01
    assert groups[1]["weight_decay"] == 0.0
    # only dense.weight is decayed; dense.bias, LayerNorm.weight/bias are not
    assert len(groups[0]["params"]) == 1
    assert len(groups[1]["params"]) == 3


def test_seed_everything_makes_draws_reproducible():
    T.seed_everything(0)
    a = torch.randn(5)
    T.seed_everything(0)
    b = torch.randn(5)
    assert torch.equal(a, b)


# --------------------------------------------------------------------------
# slow — real encoder on CPU
# --------------------------------------------------------------------------

ENCODER = "mbert-base"


@pytest.mark.slow
def test_the_loop_overfits_a_tiny_subset():
    """CLAUDE2.md T-203's done criterion: >0.95 train macro-F1 on a handful."""
    result = T.overfit_subset(ENCODER, 2, n=32, epochs=25, device="cpu")
    assert result.history[-1]["train_macro_f1"] > 0.95


@pytest.mark.slow
def test_two_runs_of_one_config_are_identical():
    a = T.overfit_subset(ENCODER, 2, n=16, epochs=3, device="cpu")
    b = T.overfit_subset(ENCODER, 2, n=16, epochs=3, device="cpu")
    assert a.config_hash == b.config_hash
    assert a.history == b.history
    assert a.best_dev_macro_f1 == b.best_dev_macro_f1


def _tiny_dataset():
    from src.data import SplitDataset, get_tokenizer, load_native

    frame = (
        load_native(2, "hin")
        .groupby("label_id", sort=True)
        .head(8)
        .reset_index(drop=True)
    )
    return SplitDataset(frame, get_tokenizer(ENCODER), max_len=64)


@pytest.mark.slow
def test_training_resumes_from_a_checkpoint(tmp_path):
    """An interrupted run continues from its checkpoint, same config."""
    from src.data import num_labels

    dataset = _tiny_dataset()
    cfg = T.TrainConfig(encoder=ENCODER, num_labels=num_labels(2), epochs=4, patience=99, seed=0)

    class Interrupt:
        def __init__(self):
            self.messages: list[str] = []

        def __call__(self, msg):
            self.messages.append(msg)
            # let epochs 0 and 1 finish (and checkpoint), then die before epoch 2's
            if msg.startswith("epoch 2:"):
                raise RuntimeError("simulated interruption")

    stopper = Interrupt()
    with pytest.raises(RuntimeError, match="simulated interruption"):
        T.train(
            T.build_model(cfg, "cpu"), dataset, dataset, cfg,
            device="cpu", work_dir=tmp_path, progress=stopper,
        )
    blob = __import__("torch").load(tmp_path / "checkpoint.pt", weights_only=False)
    assert blob["epoch"] == 1  # only epochs 0 and 1 were checkpointed

    messages: list[str] = []
    resumed = T.train(
        T.build_model(cfg, "cpu"), dataset, dataset, cfg,
        device="cpu", work_dir=tmp_path, progress=messages.append,
    )
    assert any("resumed from epoch 2" in m for m in messages)
    assert resumed.epochs_run == 4
    assert len(resumed.history) == 4
    assert [row["epoch"] for row in resumed.history] == [0, 1, 2, 3]


@pytest.mark.slow
def test_a_checkpoint_for_another_config_is_refused(tmp_path):
    from src.data import SplitDataset, get_tokenizer, load_native, num_labels

    frame = (
        load_native(2, "hin")
        .groupby("label_id", sort=True)
        .head(8)
        .reset_index(drop=True)
    )
    dataset = SplitDataset(frame, get_tokenizer(ENCODER), max_len=64)
    cfg = T.TrainConfig(encoder=ENCODER, num_labels=num_labels(2), epochs=1, patience=99, seed=0)
    T.train(T.build_model(cfg, "cpu"), dataset, dataset, cfg, device="cpu", work_dir=tmp_path)

    other = replace(cfg, lr=cfg.lr * 2, epochs=2)
    with pytest.raises(ValueError, match="written for config"):
        T.train(T.build_model(other, "cpu"), dataset, dataset, other, device="cpu", work_dir=tmp_path)
