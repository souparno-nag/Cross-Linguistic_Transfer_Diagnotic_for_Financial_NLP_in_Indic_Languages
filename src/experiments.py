"""The run log (CLAUDE2.md Outputs).

``experiments.csv`` is the one sanctioned CSV in the project: it records runs,
not corpus data, so `hard rule 2` does not apply. Every row carries the run's
config hash, so a number can always be traced back to the settings that made it.

Columns are fixed by the brief:

    run_id, encoder, split, seed, config_hash, accuracy, macro_f1, date
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from .download_dataset.paths import REPO_ROOT

RESULTS_PATH = REPO_ROOT / "experiments.csv"
FIELDNAMES = [
    "run_id",
    "encoder",
    "split",
    "seed",
    "config_hash",
    "accuracy",
    "macro_f1",
    "date",
]


def log_run(
    run_config,
    run_id: str,
    metrics: dict,
    *,
    path: str | Path = RESULTS_PATH,
) -> dict:
    """Append one result row and return it.

    ``run_config`` is a :class:`src.config.RunConfig` (duck-typed here to avoid a
    circular import); ``metrics`` needs ``accuracy`` and ``macro_f1``.
    """
    path = Path(path)
    row = {
        "run_id": run_id,
        "encoder": run_config.encoder,
        "split": run_config.split_id(),
        "seed": run_config.seed,
        "config_hash": run_config.hash(),
        "accuracy": round(float(metrics["accuracy"]), 6),
        "macro_f1": round(float(metrics["macro_f1"]), 6),
        "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    return row


def read_results(path: str | Path = RESULTS_PATH) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
