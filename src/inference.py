"""Zero-shot inference runner (CLAUDE3.md T-301).

Loads a frozen Phase 2 checkpoint and runs it over any split with **zero**
parameter updates. This is the shared runner the rest of Phase 3 builds on —
T-302's per-instance log and T-305's full condition sweep both call
`run_inference` rather than touching a model or an optimiser directly.

Hard rule 1 ("zero-shot means zero updates") is enforced, not just followed by
convention: no optimiser is ever constructed in this module, `load_frozen_model`
freezes every parameter and pins the model to eval mode, and `predict` asserts
the model never leaves eval mode and never builds a gradient graph.

Hard rule 2 ("never train in this phase") is what makes checkpoints read-only
inputs here: this module only ever calls `model.load_state_dict`, never
`train()` from `src.train`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from . import checkpoints
from .config import RunConfig
from .data import SplitDataset, get_tokenizer, load_split, num_labels
from .metrics import classification_metrics
from .train import Classifier, build_model

_RUN_CONFIG_FIELDS = {f.name for f in fields(RunConfig)}


@dataclass(frozen=True)
class LoadedRun:
    """A frozen checkpoint, ready for inference only."""

    run_id: str
    run_config: RunConfig
    model: Classifier
    device: str


def load_run_config_for(run_id: str, *, root: Path | None = None) -> RunConfig:
    """The `RunConfig` a checkpoint was trained under, from its sidecar file.

    The checkpoint blob itself only carries `TrainConfig`'s hash (T-208) —
    hyperparameters, not which split or encoder produced it. `run_config.json`
    is what makes a checkpoint self-describing enough to reload for inference
    on a *different* split, which is the whole point of Phase 3.
    """
    root = root if root is not None else checkpoints.CHECKPOINT_ROOT
    path = root / run_id / checkpoints.RUN_CONFIG_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"{run_id}: no {checkpoints.RUN_CONFIG_FILE} sidecar at {path} -- "
            "cannot recover which encoder, task or split trained this checkpoint"
        )
    raw = json.loads(path.read_text())
    known = {k: v for k, v in raw.items() if k in _RUN_CONFIG_FIELDS}
    return RunConfig(**known)


def load_frozen_model(run_id: str, *, device: str = "cpu") -> LoadedRun:
    """A checkpoint's best-dev weights, loaded, frozen, and pinned to eval mode.

    Nothing downstream of this call may put the model back into `train()` mode
    or construct an optimiser against its parameters -- that would silently
    turn a zero-shot evaluation into a fine-tuning run (hard rules 1-2).
    """
    run = load_run_config_for(run_id)
    tcfg = run.train_config()
    model = build_model(tcfg, device)
    model.load_state_dict(checkpoints.best_state_dict(run_id))
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return LoadedRun(run_id=run_id, run_config=run, model=model, device=device)


@torch.no_grad()
def predict(loaded: LoadedRun, dataset: SplitDataset, *, batch_size: int = 32) -> dict:
    """Predictions and per-class probabilities for every row in `dataset`.

    Row order is not guaranteed to match `dataset`'s own order -- callers that
    need to tie a prediction back to its row use the returned `item_id`s, the
    same join key the rest of the corpus uses (§6).
    """
    model = loaded.model
    assert not model.training, "model must be in eval mode for zero-shot inference"
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=dataset.collate
    )
    item_ids: list[str] = []
    gold: list[int] = []
    pred: list[int] = []
    probs: list[list[float]] = []
    for batch in loader:
        logits = model(
            batch["input_ids"].to(loaded.device),
            batch["attention_mask"].to(loaded.device),
        )
        assert not logits.requires_grad, "zero-shot inference must not build a graph"
        batch_probs = torch.softmax(logits, dim=-1)
        pred.extend(logits.argmax(dim=-1).cpu().tolist())
        gold.extend(batch["labels"].tolist())
        probs.extend(batch_probs.cpu().tolist())
        item_ids.extend(batch["item_ids"])
    assert not model.training, "model left eval mode during inference"
    return {"item_id": item_ids, "gold": gold, "pred": pred, "probs": probs}


def run_inference(
    run_id: str,
    task: int,
    block: str,
    lang: str,
    origin: str,
    *,
    device: str = "cpu",
    batch_size: int = 32,
) -> dict:
    """Load a frozen checkpoint and predict over any split.

    `task`/`block`/`lang`/`origin` need not match the run's own training
    split -- evaluating on a *different* one is what zero-shot transfer means.
    Tokenisation uses the run's own encoder and `max_len`, so the target text
    is processed exactly as the training text was.

    Includes aggregate metrics for convenience; T-302's per-instance log,
    keyed by `item_id`, is the artefact Phase 6 actually consumes.
    """
    loaded = load_frozen_model(run_id, device=device)
    frame = load_split(task, block, lang, origin)
    tokenizer = get_tokenizer(loaded.run_config.encoder)
    dataset = SplitDataset(frame, tokenizer, max_len=loaded.run_config.max_len)
    result = predict(loaded, dataset, batch_size=batch_size)
    metrics = classification_metrics(result["gold"], result["pred"], num_labels(task))
    return {
        **result,
        "metrics": metrics,
        "run_id": run_id,
        "split": f"task_{task}/{block}/{lang}/{origin}",
    }
