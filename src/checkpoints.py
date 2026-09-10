"""Checkpoint registry and retention (CLAUDE2.md T-208).

**Naming.** One directory per run, ``checkpoints/<run_id>/``, where
``run_id = "<config-stem>_seed<N>"`` (e.g. ``task2_hin_indicbert_seed1``). It
holds ``checkpoint.pt`` (model / optimiser / scheduler / RNG / best-so-far /
history / config hash — written after every epoch by ``train.train``) and
``run_config.json`` (the resolved :class:`src.config.RunConfig` plus its hash).

**Retrieval.** ``checkpoint_path(run_id)`` and ``load(run_id)`` — a run's
checkpoint is addressable by ``run_id`` alone, which is §8's done criterion.

**Retention.** A checkpoint is kept when it is either
  * *reported* — a row in ``experiments.csv`` has its ``run_id`` and
    ``config_hash`` (a result we published and might reload), or
  * *current* — its ``config_hash`` equals the hash of the shipped YAML for that
    run today (an interrupted run we can still resume).
Anything else — a superseded config, an aborted experiment — is an orphan and
``prune`` removes it. Checkpoints are large and gitignored; ``experiments.csv``
and ``reports/`` are the durable record.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from .download_dataset.paths import REPO_ROOT
from .experiments import RESULTS_PATH, read_results

CHECKPOINT_ROOT = REPO_ROOT / "checkpoints"
CHECKPOINT_FILE = "checkpoint.pt"
RUN_CONFIG_FILE = "run_config.json"

_RUN_ID_RE = re.compile(r"^(?P<stem>.+)_seed(?P<seed>\d+)$")


def checkpoint_dir(run_id: str) -> Path:
    return CHECKPOINT_ROOT / run_id


def checkpoint_path(run_id: str) -> Path:
    return checkpoint_dir(run_id) / CHECKPOINT_FILE


def exists(run_id: str) -> bool:
    return checkpoint_path(run_id).exists()


def load(run_id: str, *, map_location: str = "cpu") -> dict:
    """The raw checkpoint blob for a run."""
    import torch

    path = checkpoint_path(run_id)
    if not path.exists():
        raise FileNotFoundError(f"no checkpoint for run_id {run_id!r} at {path}")
    return torch.load(path, map_location=map_location, weights_only=False)


def best_state_dict(run_id: str) -> dict:
    """The best-dev model weights — what Phase 3 loads to reuse a trained model."""
    blob = load(run_id)
    state = blob.get("best", {}).get("state")
    if state is None:
        raise ValueError(f"{run_id}: checkpoint has no best-weights snapshot yet")
    return state


def metadata(run_id: str) -> dict | None:
    """Summary of one checkpoint without holding onto the model tensors."""
    path = checkpoint_path(run_id)
    if not path.exists():
        return None
    blob = load(run_id)
    best = blob.get("best", {}) or {}
    history = blob.get("history", []) or []
    return {
        "run_id": run_id,
        "config_hash": blob.get("config_hash"),  # TrainConfig hash
        "run_hash": blob.get("run_hash"),  # RunConfig hash (None on pre-T-208 checkpoints)
        "last_epoch": blob.get("epoch"),
        "epochs_recorded": len(history),
        "best_epoch": best.get("epoch"),
        "best_dev_macro_f1": best.get("macro_f1"),
        "has_best_weights": best.get("state") is not None,
        "size_bytes": path.stat().st_size,
        "mtime": path.stat().st_mtime,
    }


def list_all() -> list[dict]:
    """Metadata for every ``checkpoints/<run_id>/`` that holds a checkpoint."""
    if not CHECKPOINT_ROOT.exists():
        return []
    out = []
    for child in sorted(CHECKPOINT_ROOT.iterdir()):
        if child.is_dir() and (child / CHECKPOINT_FILE).exists():
            out.append(metadata(child.name))
    return out


# --------------------------------------------------------------------------
# Retention
# --------------------------------------------------------------------------


def _current_hashes(run_id: str) -> tuple[str | None, str | None]:
    """(RunConfig hash, TrainConfig hash) of the shipped YAML for this run_id
    today — or (None, None) if the config is gone."""
    match = _RUN_ID_RE.match(run_id)
    if not match:
        return (None, None)
    from .config import TRAIN_CONFIG_DIR, load_run_config

    yaml_path = TRAIN_CONFIG_DIR / f"{match['stem']}.yaml"
    if not yaml_path.exists():
        return (None, None)
    try:
        run = load_run_config(yaml_path, seed=int(match["seed"]))
        return (run.hash(), run.train_config().hash())
    except (ValueError, OSError):
        return (None, None)


def _current_hash(run_id: str) -> str | None:
    """The current RunConfig hash for this run_id (back-compat shim for tests)."""
    return _current_hashes(run_id)[0]


def _reported_run_hashes(results_path: str | Path) -> dict[str, set[str]]:
    """run_id -> the RunConfig hashes logged for it in experiments.csv."""
    out: dict[str, set[str]] = {}
    for row in read_results(results_path):
        out.setdefault(row["run_id"], set()).add(row["config_hash"])
    return out


def classify(results_path: str | Path = RESULTS_PATH) -> dict[str, list[dict]]:
    """Split existing checkpoints into ``keep`` and ``orphan``, with the reason.

    A checkpoint identifies its run by ``run_hash`` (the RunConfig hash, which
    is also what ``experiments.csv`` records). Pre-T-208 checkpoints have no
    ``run_hash``; those fall back to matching ``config_hash`` (the TrainConfig
    hash) against the current YAML's.
    """
    reported = _reported_run_hashes(results_path)
    keep, orphan = [], []
    for meta in list_all():
        run_id = meta["run_id"]
        run_hash, cfg_hash = meta["run_hash"], meta["config_hash"]
        cur_run_hash, cur_cfg_hash = _current_hashes(run_id)

        if run_hash and run_hash in reported.get(run_id, set()):
            keep.append({**meta, "reason": "reported in experiments.csv"})
        elif run_hash and run_hash == cur_run_hash:
            keep.append({**meta, "reason": "matches current config (resumable)"})
        elif run_hash is None and cfg_hash and cfg_hash == cur_cfg_hash:
            keep.append({**meta, "reason": "legacy checkpoint, config unchanged"})
        else:
            orphan.append({**meta, "reason": "superseded config / not reported"})
    return {"keep": keep, "orphan": orphan}


def prune(results_path: str | Path = RESULTS_PATH, *, dry_run: bool = True) -> list[str]:
    """Remove orphan checkpoint dirs. Returns the run_ids removed (or that would be)."""
    orphans = [m["run_id"] for m in classify(results_path)["orphan"]]
    if not dry_run:
        for run_id in orphans:
            shutil.rmtree(checkpoint_dir(run_id))
    return orphans
