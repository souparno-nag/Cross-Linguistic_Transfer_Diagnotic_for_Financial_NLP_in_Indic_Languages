"""Tests for T-301's zero-shot inference runner.

The fast block checks the part that must hold regardless of which encoder is
loaded: a model handed to `predict` never leaves eval mode, never builds a
gradient graph, and its weights are untouched by inference -- hard rules 1-2.
The `slow` block is the acceptance criterion itself: load a real checkpoint
and confirm re-running it on its own training split reproduces the Phase 2
dev number that was recorded when it was trained.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
import torch
from torch import nn

from src import checkpoints
from src import inference as I
from src.data import SplitDataset


# --------------------------------------------------------------------------
# Fast -- no model download
# --------------------------------------------------------------------------


class _FakeTokenizer:
    """Maps each character to its codepoint; no network, no vocab file."""

    pad_token_id = 0

    def __call__(self, texts, *, truncation, max_length, padding):
        del truncation, padding
        input_ids = [[ord(c) % 97 + 1 for c in text][:max_length] for text in texts]
        attention_mask = [[1] * len(ids) for ids in input_ids]
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _TinyModel(nn.Module):
    """Stands in for `Classifier`: same call signature, tiny and download-free."""

    def __init__(self, num_labels: int = 3):
        super().__init__()
        self.embed = nn.Embedding(128, 8, padding_idx=0)
        self.classifier = nn.Linear(8, num_labels)

    def forward(self, input_ids, attention_mask):
        del attention_mask
        pooled = self.embed(input_ids).mean(dim=1)
        return self.classifier(pooled)


def _tiny_dataset(n: int = 6, num_labels: int = 3) -> SplitDataset:
    frame = pd.DataFrame(
        {
            "text": [f"row {i} sentence" for i in range(n)],
            "label_id": [i % num_labels for i in range(n)],
            "item_id": [f"it{i}" for i in range(n)],
        }
    )
    return SplitDataset(frame, _FakeTokenizer(), max_len=32)


def test_predict_keeps_the_model_in_eval_mode_and_untouched():
    model = _TinyModel()
    model.eval()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    loaded = I.LoadedRun(run_id="fake", run_config=None, model=model, device="cpu")

    result = I.predict(loaded, _tiny_dataset(), batch_size=4)

    assert not model.training
    after = model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before), "weights moved"
    assert len(result["item_id"]) == 6
    assert len(result["pred"]) == 6
    assert all(len(row) == 3 for row in result["probs"])


def test_predict_refuses_a_model_left_in_train_mode():
    model = _TinyModel()
    model.train()  # deliberately wrong: predict should not silently fix this
    loaded = I.LoadedRun(run_id="fake", run_config=None, model=model, device="cpu")
    with pytest.raises(AssertionError, match="eval mode"):
        I.predict(loaded, _tiny_dataset(), batch_size=4)


def test_predict_builds_no_gradient_graph():
    model = _TinyModel()
    model.eval()
    loaded = I.LoadedRun(run_id="fake", run_config=None, model=model, device="cpu")
    I.predict(loaded, _tiny_dataset(), batch_size=4)
    # no leaked .grad on any parameter -- nothing ever called .backward()
    assert all(p.grad is None for p in model.parameters())


def test_load_run_config_for_reads_the_sidecar(tmp_path):
    from src.config import RunConfig

    run = RunConfig(encoder="mbert-base", task=2, lang="hin", epochs=1, seed=0)
    run_dir = tmp_path / "some_run_seed0"
    run_dir.mkdir()
    (run_dir / checkpoints.RUN_CONFIG_FILE).write_text(json.dumps(run.to_dict()))

    loaded = I.load_run_config_for("some_run_seed0", root=tmp_path)
    assert loaded.encoder == "mbert-base"
    assert loaded.task == 2
    assert loaded.hash() == run.hash()


def test_load_run_config_for_raises_when_no_sidecar_exists(tmp_path):
    with pytest.raises(FileNotFoundError, match="no run_config.json"):
        I.load_run_config_for("does_not_exist", root=tmp_path)


# --------------------------------------------------------------------------
# slow -- real encoder, a real checkpoint, the acceptance criterion
# --------------------------------------------------------------------------

ENCODER = "mbert-base"


@pytest.mark.slow
def test_reevaluating_on_the_training_splits_dev_fold_reproduces_the_recorded_number(
    tmp_path, monkeypatch
):
    """T-301's done criterion: re-running on the training split reproduces the
    Phase 2 dev number recorded at training time."""
    from src.config import RunConfig, run_training

    run = RunConfig(
        encoder=ENCODER,
        task=3,
        lang="hin",
        max_len=64,
        batch_size=32,
        epochs=1,
        patience=1,
        dev_fraction=0.2,
        seed=0,
    )
    work_dir = tmp_path / "checkpoints" / "t301_check_seed0"
    result = run_training(run, device="cpu", work_dir=work_dir)

    monkeypatch.setattr(checkpoints, "CHECKPOINT_ROOT", tmp_path / "checkpoints")
    loaded = I.load_frozen_model("t301_check_seed0", device="cpu")
    assert not loaded.model.training

    from src.data import SplitDataset, get_tokenizer, load_split, stratified_split

    frame = load_split(run.task, run.resolved_block(), run.lang, run.origin)
    parts = stratified_split(frame, seed=run.seed, dev=run.dev_fraction)
    dataset = SplitDataset(parts["dev"], get_tokenizer(run.encoder), max_len=run.max_len)

    from src.metrics import classification_metrics

    prediction = I.predict(loaded, dataset, batch_size=32)
    metrics = classification_metrics(
        prediction["gold"], prediction["pred"], num_labels=len(frame["label_id"].unique())
    )
    assert metrics["macro_f1"] == pytest.approx(result["best_dev_macro_f1"], abs=1e-6)


@pytest.mark.slow
def test_run_inference_end_to_end_on_a_different_split(tmp_path, monkeypatch):
    """T-301's other half: run over *any* split, not just the training one."""
    from src.config import RunConfig, run_training

    run = RunConfig(
        encoder=ENCODER, task=3, lang="hin", max_len=64, epochs=1, patience=1,
        dev_fraction=0.2, seed=0,
    )
    work_dir = tmp_path / "checkpoints" / "t301_cross_seed0"
    run_training(run, device="cpu", work_dir=work_dir)
    monkeypatch.setattr(checkpoints, "CHECKPOINT_ROOT", tmp_path / "checkpoints")

    out = I.run_inference("t301_cross_seed0", 3, "B", "ben", "native", device="cpu")
    assert set(out["item_id"]) and len(out["pred"]) == len(out["gold"])
    assert 0.0 <= out["metrics"]["macro_f1"] <= 1.0
    assert out["split"] == "task_3/B/ben/native"
