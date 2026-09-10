"""T-114 — the datasheet for corpus v1.0.

Every number here is read back out of the artefacts that produced it: the
frozen manifests, T-107's integrity report, T-109's drift table, T-110's
calibration verdict, T-111's ranking. Nothing is typed in by hand.

That is the whole design. A datasheet whose figures are transcribed drifts from
the data the first time anything is regenerated, and it drifts silently,
because nobody re-reads a document that looks finished. Reading them back means
the datasheet is either correct or it fails to build.

Structure follows Gebru et al., *Datasheets for Datasets* — motivation,
composition, collection, preprocessing, uses, distribution, maintenance —
because that is the form the field expects, with §8's required content mapped
onto it.
"""

from __future__ import annotations

import json

import pandas as pd

from .corpus_io import CONFIG_DIR, is_numeral_task
from .download_dataset.paths import REPO_ROOT
from .freeze import frozen_dir
from .ids import BLOCK_NATIVE_LANG, targets_for_block

REPORT_ROOT = REPO_ROOT / "reports"
TASKS = (1, 2, 3)
TASK_CONTENT = {
    1: "numerals in financial text (span annotation, no class label)",
    2: "sustainability sentences (binary classification)",
    3: "ESG news headlines (10-class classification)",
}


def load_artefacts(task: int) -> dict:
    """Everything the datasheet says about one task, from where it was written."""
    release = frozen_dir(task)
    manifest_path = release / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"task {task} is not frozen ({manifest_path} missing); "
            "run `python -m scripts.t112_freeze --task {task}` first"
        )
    reports = REPORT_ROOT / f"task_{task}"
    ranking = pd.read_parquet(reports / "direction_ranking.parquet")
    return {
        "manifest": json.loads(manifest_path.read_text()),
        "integrity": json.loads((reports / "integrity.json").read_text()),
        "calibration": json.loads((reports / "tau_calibration.json").read_text()),
        "ranking": ranking,
        "drift": pd.read_parquet(reports / "drift_by_direction.parquet"),
    }


def direction_table(artefacts: dict, task: int) -> pd.DataFrame:
    """Drift and entity preservation per direction — §8's two required rates."""
    columns = {
        "src_lang": "source",
        "tgt_lang": "target",
        "quadrant": "typology",
        "median_sim": "median LaBSE",
        "drift_rate": "below τ",
        "entity_preserved": "entities kept",
        "corrupt_rate": "corrupted",
    }
    if is_numeral_task(task):
        columns["span_recovery"] = "numeral re-found"
    frame = artefacts["ranking"][list(columns)].rename(columns=columns)
    for column in ("below τ", "entities kept", "corrupted", "numeral re-found"):
        if column in frame:
            frame[column] = (frame[column] * 100).round(1).astype(str) + "%"
    frame["median LaBSE"] = frame["median LaBSE"].round(3)
    return frame


def totals() -> dict:
    """Row and split counts across every frozen task."""
    rows = splits = 0
    for task in TASKS:
        manifest = json.loads((frozen_dir(task) / "manifest.json").read_text())
        rows += manifest["rows"]
        splits += manifest["splits"]
    return {"rows": rows, "splits": splits}
