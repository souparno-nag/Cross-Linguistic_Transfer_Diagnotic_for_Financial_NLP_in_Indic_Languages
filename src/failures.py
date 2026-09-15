"""Export the failed-instance set (CLAUDE3.md T-307).

T-306's predicate (`ŷ_src == y AND ŷ_tgt != y`) picks out items the model
answered correctly one language earlier and got wrong here -- ruling out
item difficulty and model incapacity by construction, which is what makes
this the set Phase 6's diagnostics can actually attribute to something. This
module just persists that set: `export_failures` joins a source (in-language
ceiling) and target prediction log, filters to the mismatch rows, and writes
them to Parquet.

Path carries a `task_{n}` level for the same reason `data/predictions/` does
(T-302, CLAUDE.md §5): task 2 and task 3 both produce a condition named
`transfer_hin_to_ben`, and without the level the two tasks' failure sets
would collide and merge.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .download_dataset.paths import REPO_ROOT
from .mismatch import filter_mismatches, join_source_target

FAILURES_ROOT = REPO_ROOT / "data" / "failures"


def failure_path(
    task: int,
    condition_id: str,
    *,
    root: Path | None = None,
    encoder: str | None = None,
) -> Path:
    """`data/failures/task_{n}/[{encoder}/]{condition_id}.parquet`.

    `encoder` namespaces the file per CLAUDE5.md rule 1. Unlike the prediction
    log, `export_failures` *overwrites* rather than merging, so without this a
    second encoder's run would not merely mix rows in — it would delete the
    first encoder's failure set outright, which is Phase 6's input.

    `encoder=None` keeps the original path, where IndicBERT's committed sets
    live and where `src.diagnostics` reads them from. See `predictions.log_path`
    for why the default encoder is not moved into a level of its own.
    """
    root = root if root is not None else FAILURES_ROOT
    base = root / f"task_{task}"
    if encoder is not None:
        base = base / encoder
    return base / f"{condition_id}.parquet"


def export_failures(
    task: int,
    condition_id: str,
    source: pd.DataFrame,
    target: pd.DataFrame,
    *,
    root: Path | None = None,
    encoder: str | None = None,
) -> tuple[Path, pd.DataFrame]:
    """Join, filter to mismatches, and write. Returns the path and the frame
    written (empty if the condition has zero mismatches -- writing an empty
    file is still correct: it records that the check ran and found none,
    rather than leaving silence indistinguishable from "not yet run")."""
    joined = join_source_target(source, target)
    failures = filter_mismatches(joined)
    path = failure_path(task, condition_id, root=root, encoder=encoder)
    path.parent.mkdir(parents=True, exist_ok=True)
    failures.to_parquet(path, index=False)
    return path, failures


def read_failures(
    task: int, condition_id: str, *, root: Path | None = None, encoder: str | None = None
) -> pd.DataFrame:
    path = failure_path(task, condition_id, root=root, encoder=encoder)
    if not path.exists():
        raise FileNotFoundError(f"no failure set for condition {condition_id!r} at {path}")
    return pd.read_parquet(path)
