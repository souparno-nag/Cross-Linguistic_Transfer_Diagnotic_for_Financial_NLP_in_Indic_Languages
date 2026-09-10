"""Fine-tuning loop for the in-language baselines (CLAUDE2.md T-203).

The architecture is fixed by the brief: the encoder's final hidden state at the
first position (``[CLS]``) → one ``nn.Linear`` → cross-entropy. No extra dense
layer, no alternative pooling, no class weighting or other loss trick — those
would make the baseline something other than "the published monolingual model,
reproduced".

The loop is deterministic (`hard rule 5`): a config and a seed pin every random
draw, so two runs log identical metrics. It checkpoints after each epoch and
resumes from that checkpoint, because GPU access is intermittent (CLAUDE.md §3)
and a run that must restart from zero is not survivable.

Early stopping watches dev **macro**-F1 (`hard rule 6`); the best weights are
restored before the result is returned.
"""

from __future__ import annotations

import json
import math
import os
import random
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

# The 1B translation model needed this on the 4 GB card; a fine-tune with a
# large batch hits the same fragmentation OOM (T-205). Must precede torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, get_linear_schedule_with_warmup

from .audit import config_hash as _hash_dict
from .data import SplitDataset, resolve_encoder
from .metrics import classification_metrics


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class TrainConfig:
    """Everything that can change a metric. Hashed into every artefact.

    Batch size is included because on the 4 GB card it interacts with
    ``grad_accum``; the effective batch is ``batch_size * grad_accum`` and that
    *does* move results, so both are pinned.
    """

    encoder: str
    num_labels: int
    max_len: int = 128
    batch_size: int = 16
    grad_accum: int = 1
    lr: float = 2e-5
    weight_decay: float = 0.01
    epochs: int = 10
    warmup_ratio: float = 0.1
    patience: int = 3
    seed: int = 0
    fp16: bool = False

    def hash(self) -> str:
        return _hash_dict(asdict(self))


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_macro_f1: float
    dev_macro_f1: float
    dev_accuracy: float


@dataclass
class TrainResult:
    config_hash: str
    best_dev_macro_f1: float
    best_epoch: int
    epochs_run: int
    stopped_early: bool
    history: list[dict] = field(default_factory=list)


# --------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------
# Model — the whole architecture
# --------------------------------------------------------------------------


class Classifier(nn.Module):
    def __init__(self, encoder_hf_id: str, num_labels: int):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_hf_id)
        hidden = self.encoder.config.hidden_size
        self.classifier = nn.Linear(hidden, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0]  # [CLS] position, final layer
        return self.classifier(cls)


def build_model(config: TrainConfig, device: str) -> Classifier:
    hf_id = resolve_encoder(config.encoder).hf_id
    # Seed before construction: the classifier head is randomly initialised, and
    # if that draw is not pinned then two runs of one config start from
    # different weights and diverge (`hard rule 5`).
    seed_everything(config.seed)
    model = Classifier(hf_id, config.num_labels)
    return model.to(device)


# --------------------------------------------------------------------------
# Data plumbing
# --------------------------------------------------------------------------


def _loader(dataset: SplitDataset, config: TrainConfig, *, shuffle: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        collate_fn=dataset.collate,
        generator=generator,
        num_workers=0,
    )


def evaluate_dataset(
    model: Classifier, dataset: SplitDataset, config: "TrainConfig", device: str
) -> dict:
    """Metrics for a whole dataset, using the config's batch size."""
    return evaluate(
        model, _loader(dataset, config, shuffle=False), device, config.num_labels
    )


@torch.no_grad()
def evaluate(model: Classifier, loader: DataLoader, device: str, num_labels: int) -> dict:
    model.eval()
    y_true: list[int] = []
    y_pred: list[int] = []
    for batch in loader:
        logits = model(
            batch["input_ids"].to(device), batch["attention_mask"].to(device)
        )
        y_pred.extend(logits.argmax(dim=-1).cpu().tolist())
        y_true.extend(batch["labels"].tolist())
    return classification_metrics(y_true, y_pred, num_labels)


# --------------------------------------------------------------------------
# Checkpointing
# --------------------------------------------------------------------------


def _save_checkpoint(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    tmp.replace(path)  # atomic: a crash mid-write cannot corrupt the resume point


def _restore_rng(rng: dict) -> None:
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"])
    if rng.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng["cuda"])


def _capture_rng() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------


def _param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """Standard AdamW grouping: no decay on biases and LayerNorm weights."""
    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    decay, plain = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (plain if any(nd in name for nd in no_decay) else decay).append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": plain, "weight_decay": 0.0},
    ]


def train(
    model: Classifier,
    train_ds: SplitDataset,
    dev_ds: SplitDataset,
    config: TrainConfig,
    *,
    device: str,
    work_dir: str | Path | None = None,
    resume: bool = True,
    progress=None,
) -> TrainResult:
    """Fine-tune ``model`` and return its best dev macro-F1.

    ``work_dir`` gets ``checkpoint.pt`` after every epoch; if it already holds
    one and ``resume`` is set, training continues from it. ``progress`` is an
    optional ``callable(str)`` for a status line.
    """
    say = progress or (lambda _msg: None)
    seed_everything(config.seed)

    train_loader = _loader(train_ds, config, shuffle=True)
    dev_loader = _loader(dev_ds, config, shuffle=False)

    steps_per_epoch = math.ceil(len(train_loader) / config.grad_accum)
    total_steps = steps_per_epoch * config.epochs
    optimizer = torch.optim.AdamW(
        _param_groups(model, config.weight_decay), lr=config.lr
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(config.warmup_ratio * total_steps),
        num_training_steps=total_steps,
    )
    use_amp = config.fp16 and device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = nn.CrossEntropyLoss()

    ckpt_path = Path(work_dir) / "checkpoint.pt" if work_dir else None
    start_epoch = 0
    best = {"macro_f1": -1.0, "epoch": -1, "state": None}
    history: list[EpochMetrics] = []
    since_improve = 0

    if ckpt_path and resume and ckpt_path.exists():
        blob = torch.load(ckpt_path, map_location=device, weights_only=False)
        if blob["config_hash"] != config.hash():
            raise ValueError(
                f"checkpoint at {ckpt_path} was written for config "
                f"{blob['config_hash']}, not {config.hash()}"
            )
        model.load_state_dict(blob["model"])
        optimizer.load_state_dict(blob["optimizer"])
        scheduler.load_state_dict(blob["scheduler"])
        scaler.load_state_dict(blob["scaler"])
        _restore_rng(blob["rng"])
        start_epoch = blob["epoch"] + 1
        best = blob["best"]
        since_improve = blob["since_improve"]
        history = [EpochMetrics(**m) for m in blob["history"]]
        say(f"resumed from epoch {start_epoch} ({ckpt_path})")

    for epoch in range(start_epoch, config.epochs):
        model.train()
        running = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(
                    batch["input_ids"].to(device),
                    batch["attention_mask"].to(device),
                )
                loss = loss_fn(logits, batch["labels"].to(device)) / config.grad_accum
            scaler.scale(loss).backward()
            running += loss.item() * config.grad_accum
            if (step + 1) % config.grad_accum == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        train_metrics = evaluate(model, _loader(train_ds, config, shuffle=False), device, config.num_labels)
        dev_metrics = evaluate(model, dev_loader, device, config.num_labels)
        row = EpochMetrics(
            epoch=epoch,
            train_loss=running / len(train_loader),
            train_macro_f1=train_metrics["macro_f1"],
            dev_macro_f1=dev_metrics["macro_f1"],
            dev_accuracy=dev_metrics["accuracy"],
        )
        history.append(row)
        say(
            f"epoch {epoch}: loss {row.train_loss:.4f} "
            f"train_f1 {row.train_macro_f1:.4f} dev_f1 {row.dev_macro_f1:.4f}"
        )

        if row.dev_macro_f1 > best["macro_f1"]:
            best = {
                "macro_f1": row.dev_macro_f1,
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
            since_improve = 0
        else:
            since_improve += 1

        if ckpt_path:
            _save_checkpoint(
                ckpt_path,
                {
                    "config_hash": config.hash(),
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "rng": _capture_rng(),
                    "best": best,
                    "since_improve": since_improve,
                    "history": [asdict(m) for m in history],
                },
            )

        if since_improve >= config.patience:
            say(f"early stop: no dev improvement for {config.patience} epochs")
            break

    stopped_early = since_improve >= config.patience
    if best["state"] is not None:
        model.load_state_dict(best["state"])

    return TrainResult(
        config_hash=config.hash(),
        best_dev_macro_f1=best["macro_f1"],
        best_epoch=best["epoch"],
        epochs_run=history[-1].epoch + 1 if history else 0,
        stopped_early=stopped_early,
        history=[asdict(m) for m in history],
    )


# --------------------------------------------------------------------------
# T-203 acceptance: overfit a tiny subset
# --------------------------------------------------------------------------


def overfit_subset(
    encoder: str,
    task: int,
    *,
    n: int = 48,
    epochs: int = 30,
    lr: float = 5e-5,
    device: str = "cpu",
    progress=None,
) -> TrainResult:
    """Train on ~``n`` class-balanced rows until the model memorises them.

    A model that cannot reach >0.95 **train** macro-F1 on this handful has a
    wiring bug — wrong pooling, a frozen encoder, a label mismatch. This is a
    correctness check, not a baseline (CLAUDE2.md T-203).

    The subset is balanced across classes because macro-F1 on an all-one-class
    sample is capped well below 1.0 however well the model fits it, which would
    fail the check for the wrong reason.
    """
    from .data import get_tokenizer, load_native, num_labels

    k = num_labels(task)
    full = load_native(task, "hin")
    per_class = max(1, n // k)
    frame = full.groupby("label_id", sort=True).head(per_class).reset_index(drop=True)
    tokenizer = get_tokenizer(encoder)
    dataset = SplitDataset(frame, tokenizer, max_len=128)

    config = TrainConfig(
        encoder=encoder,
        num_labels=num_labels(task),
        batch_size=16,
        lr=lr,
        epochs=epochs,
        warmup_ratio=0.1,
        patience=epochs,  # never early-stop; run the full budget
        seed=0,
    )
    model = build_model(config, device)
    # dev == train: "did it memorise the training set" is the whole question.
    return train(model, dataset, dataset, config, device=device, work_dir=None, progress=progress)
