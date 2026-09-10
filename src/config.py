"""Run configuration: YAML in, a hashed spec and an executable run out (T-204).

One YAML file per baseline (`configs/train/*.yaml`) holds everything that
defines a run — which split, which encoder, and the training hyperparameters.
:func:`load_run_config` reads it into a frozen :class:`RunConfig`; the config's
hash is stamped into every ``experiments.csv`` row so a metric is always
traceable to the exact settings that produced it (`hard rule 4`).

`RunConfig` is deliberately a superset of `train.TrainConfig`: the training
loop only needs the hyperparameters, but a *metric* also depends on which rows
were trained on and how they were split, so the run hash covers those too.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import yaml

from .audit import config_hash as _hash_dict
from .data import num_labels
from .download_dataset.paths import REPO_ROOT
from .ids import block_for_native_lang
from .train import TrainConfig, build_model, evaluate_dataset
from .train import train as _train
from .unicode_ranges import normalise_lang

TRAIN_CONFIG_DIR = REPO_ROOT / "configs" / "train"

_INT_FIELDS = {"task", "max_len", "batch_size", "grad_accum", "epochs", "patience", "seed"}
_FLOAT_FIELDS = {"lr", "weight_decay", "warmup_ratio", "dev_fraction", "test_fraction"}
_BOOL_FIELDS = {"fp16"}


@dataclass(frozen=True)
class RunConfig:
    """A complete, reproducible training run."""

    encoder: str
    task: int
    lang: str
    origin: str = "native"
    block: str | None = None
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
    dev_fraction: float = 0.1
    test_fraction: float = 0.0

    def __post_init__(self):
        if self.origin not in ("native", "mt"):
            raise ValueError(f"origin must be native|mt, got {self.origin!r}")
        if not 0.0 < self.dev_fraction < 1.0:
            raise ValueError(
                "dev_fraction must be in (0, 1): early stopping needs a dev set"
            )
        if not 0.0 <= self.test_fraction < 1.0:
            raise ValueError(f"test_fraction out of range: {self.test_fraction}")
        if self.dev_fraction + self.test_fraction >= 1.0:
            raise ValueError("dev_fraction + test_fraction leaves no training data")
        # resolve_block raises for a language with no native block (Malayalam)
        # unless one was given explicitly.
        self.resolved_block()
        # num_labels raises for the numeral task (1), which has no label.
        num_labels(self.task)

    # -- derived -------------------------------------------------------------

    def resolved_block(self) -> str:
        if self.block is not None:
            return self.block
        return block_for_native_lang(normalise_lang(self.lang))

    def split_id(self) -> str:
        return f"task{self.task}/{self.resolved_block()}/{normalise_lang(self.lang)}/{self.origin}"

    def train_config(self) -> TrainConfig:
        return TrainConfig(
            encoder=self.encoder,
            num_labels=num_labels(self.task),
            max_len=self.max_len,
            batch_size=self.batch_size,
            grad_accum=self.grad_accum,
            lr=self.lr,
            weight_decay=self.weight_decay,
            epochs=self.epochs,
            warmup_ratio=self.warmup_ratio,
            patience=self.patience,
            seed=self.seed,
            fp16=self.fp16,
        )

    def hash(self) -> str:
        """Stable hash of the *resolved* run — equivalent specs hash equal."""
        payload = asdict(self)
        payload["block"] = self.resolved_block()
        payload["lang"] = normalise_lang(self.lang)
        return _hash_dict(payload)

    def to_dict(self) -> dict:
        return {**asdict(self), "block": self.resolved_block(), "hash": self.hash()}


_FIELD_NAMES = {f.name for f in fields(RunConfig)}


def _coerce(raw: dict) -> dict:
    """Fix YAML's type surprises — notably ``2e-5`` parsing as a string."""
    out = dict(raw)
    for key in _INT_FIELDS & out.keys():
        out[key] = int(out[key])
    for key in _FLOAT_FIELDS & out.keys():
        out[key] = float(out[key])
    for key in _BOOL_FIELDS & out.keys():
        out[key] = bool(out[key])
    return out


def load_run_config(path: str | Path, **overrides) -> RunConfig:
    """Read a YAML run config, applying keyword overrides (e.g. ``seed=1``).

    T-206 sweeps seeds by overriding this rather than keeping three near-identical
    files.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path} is not a mapping")
    merged = {**raw, **overrides}
    unknown = set(merged) - _FIELD_NAMES
    if unknown:
        raise ValueError(
            f"{path}: unknown keys {sorted(unknown)}; allowed {sorted(_FIELD_NAMES)}"
        )
    return RunConfig(**_coerce(merged))


# --------------------------------------------------------------------------
# Executing a config
# --------------------------------------------------------------------------


def run_training(
    run: RunConfig,
    *,
    device: str = "cpu",
    work_dir: str | Path | None = None,
    progress=None,
) -> dict:
    """Load the split, train, and evaluate — the unit T-206 repeats per seed.

    Training uses the ``train`` fold; early stopping watches ``dev``; the
    reported numbers are from ``test`` when ``test_fraction > 0``, otherwise from
    ``dev``. Deterministic: same config, same metrics.
    """
    from .data import (
        SplitDataset,
        get_tokenizer,
        load_split,
        stratified_split,
    )

    frame = load_split(run.task, run.resolved_block(), run.lang, run.origin)
    parts = stratified_split(
        frame, seed=run.seed, dev=run.dev_fraction, test=run.test_fraction
    )
    tokenizer = get_tokenizer(run.encoder)
    datasets = {
        name: SplitDataset(part, tokenizer, max_len=run.max_len)
        for name, part in parts.items()
    }

    tcfg = run.train_config()
    model = build_model(tcfg, device)
    if work_dir is not None:
        Path(work_dir).mkdir(parents=True, exist_ok=True)
        (Path(work_dir) / "run_config.json").write_text(
            json.dumps(run.to_dict(), indent=2) + "\n"
        )

    result = _train(
        model,
        datasets["train"],
        datasets["dev"],
        tcfg,
        device=device,
        work_dir=work_dir,
        progress=progress,
    )

    eval_name = "test" if "test" in datasets else "dev"
    eval_metrics = evaluate_dataset(model, datasets[eval_name], tcfg, device)
    return {
        "config_hash": run.hash(),
        "train_config_hash": tcfg.hash(),
        "split": run.split_id(),
        "eval_fold": eval_name,
        "n_train": len(datasets["train"]),
        "n_dev": len(datasets["dev"]),
        "n_eval": len(datasets[eval_name]),
        "accuracy": eval_metrics["accuracy"],
        "macro_f1": eval_metrics["macro_f1"],
        "per_class_f1": eval_metrics["per_class_f1"],
        "best_dev_macro_f1": result.best_dev_macro_f1,
        "best_epoch": result.best_epoch,
        "epochs_run": result.epochs_run,
        "stopped_early": result.stopped_early,
        "history": result.history,
    }
