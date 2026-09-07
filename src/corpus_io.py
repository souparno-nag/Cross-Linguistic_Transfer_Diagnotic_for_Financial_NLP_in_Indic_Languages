"""Every read and write of corpus data (§5).

No other module calls ``pd.read_parquet`` or ``to_parquet``. Routing all
access through here is what keeps the format rules enforceable rather than
merely stated: Parquet only, never CSV (§4 rule 2), never pickle (rule 3), the
exact column set in §6, and labels drawn only from ``configs/labels.json``
(rule 6).

Two schemas exist. Classification tasks carry `label`/`label_id`; task 1 marks
a number inside a sentence and carries the numeral columns of §6.1 instead.
"""

from __future__ import annotations

import json

import pandas as pd

from .download_dataset.paths import REPO_ROOT
from .ids import validate_keys

CONFIG_DIR = REPO_ROOT / "configs"
LABELS_PATH = CONFIG_DIR / "labels.json"
CORPUS_ROOT = REPO_ROOT / "data" / "raw"
FROZEN_ROOT = REPO_ROOT / "data" / "v1.0"

# §6, in order. `label`/`label_id` are swapped for the §6.1 numeral columns on
# task 1.
BASE_BEFORE_LABEL = ["block_id", "item_id", "lang", "origin", "src_lang", "text"]
BASE_AFTER_LABEL = ["labse_sim", "flags"]
CLASSIFICATION_COLUMNS = ["label", "label_id"]
NUMERAL_COLUMNS = [
    "number_indic",
    "number_english",
    "start_posn",
    "end_posn",
    "magnitude",
    "span_recovered",
]

NUMERAL_TASKS = (1,)
ORIGINS = ("native", "mt")


def is_numeral_task(task: int) -> bool:
    return task in NUMERAL_TASKS


def schema_columns(task: int) -> list[str]:
    middle = NUMERAL_COLUMNS if is_numeral_task(task) else CLASSIFICATION_COLUMNS
    return BASE_BEFORE_LABEL + middle + BASE_AFTER_LABEL


def split_path(task: int, block: str, lang: str, frozen: bool = False) -> "object":
    """`data/raw/task_2/H/hin.parquet`.

    The `task_{n}` level is not decoration: without it the three corpora
    collide on identical block/language paths and one silently overwrites
    another (§5).
    """
    root = FROZEN_ROOT if frozen else CORPUS_ROOT
    return root / f"task_{task}" / block / f"{lang}.parquet"


# --------------------------------------------------------------------------
# Label schema
# --------------------------------------------------------------------------


def load_labels(task: int) -> dict:
    """Canonical labels for a task, or ``None`` for the numeral task."""
    if not LABELS_PATH.exists():
        raise FileNotFoundError(
            f"{LABELS_PATH} does not exist; build it with scripts/t103_ingest.py"
        )
    schema = json.loads(LABELS_PATH.read_text())
    entry = schema.get(f"task_{task}")
    if entry is None:
        return None
    return {"labels": entry["labels"], "index": {v: i for i, v in enumerate(entry["labels"])}}


def write_labels(schema: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    LABELS_PATH.write_text(json.dumps(schema, indent=2, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def validate(frame: pd.DataFrame, task: int) -> None:
    """Everything §6 and §4 require of a corpus frame. Raises on any breach."""
    expected = schema_columns(task)
    if list(frame.columns) != expected:
        missing = [c for c in expected if c not in frame.columns]
        extra = [c for c in frame.columns if c not in expected]
        raise ValueError(
            f"task {task} schema mismatch. missing={missing} unexpected={extra} "
            f"order_ok={list(frame.columns) == expected}"
        )

    bad_origin = set(frame["origin"]) - set(ORIGINS)
    if bad_origin:
        raise ValueError(f"unknown origin values: {sorted(bad_origin)}")

    # A native row descends from nothing; an MT row must say what it came from.
    natives = frame[frame["origin"] == "native"]
    if natives["src_lang"].notna().any():
        raise ValueError("native rows must have a null src_lang")
    machine = frame[frame["origin"] == "mt"]
    if machine["src_lang"].isna().any():
        raise ValueError("mt rows must record src_lang")
    if natives["labse_sim"].notna().any():
        raise ValueError("native rows must have a null labse_sim (§6)")

    if frame["text"].isna().any():
        raise ValueError("null text is never valid; flag empty output instead")

    if not is_numeral_task(task):
        schema = load_labels(task)
        if schema is None:
            raise ValueError(f"task {task} is a classification task but has no labels")
        unknown = sorted(set(frame["label"]) - set(schema["labels"]))
        if unknown:
            # Rule 6: no coercing, no dropping, no inventing.
            raise ValueError(
                f"labels outside configs/labels.json: {unknown}. "
                "Labels are never invented, remapped or reordered — fix the "
                "source or the schema deliberately."
            )
            
        expected_ids = frame["label"].map(schema["index"])
        if not (frame["label_id"] == expected_ids).all():
            wrong = frame[frame["label_id"] != expected_ids]
            raise ValueError(
                f"{len(wrong)} rows have a label_id disagreeing with labels.json, "
                f"e.g. {wrong.iloc[0]['label']!r} -> {wrong.iloc[0]['label_id']}"
            )
    else:
        # A native row's offsets index the source string and must be exact. An
        # MT row's were re-derived from the translation (§6.1), so they are
        # checked only where recovery claims to have succeeded; a row whose
        # number was not found carries -1 and is flagged, never dropped.
        native_span = frame["origin"] == "native"
        recovered = frame["span_recovered"].fillna(False).astype(bool)
        spans = frame[native_span | recovered]
        broken = [
            row["item_id"]
            for _, row in spans.iterrows()
            if str(row["text"])[int(row["start_posn"]) : int(row["end_posn"])]
            != str(row["number_indic"])
        ]
        if broken:
            raise ValueError(
                f"{len(broken)} rows have offsets that do not point at their "
                f"number, e.g. {broken[:3]} — on a native row see §4 rule 9, "
                "this is what normalising does; on an MT row it is a span "
                "recovered onto the wrong characters"
            )

    validate_keys(frame)


# --------------------------------------------------------------------------
# Read / write
# --------------------------------------------------------------------------


def write_split(
    frame: pd.DataFrame, task: int, block: str, lang: str, frozen: bool = False
) -> "object":
    validate(frame, task)
    path = split_path(task, block, lang, frozen=frozen)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return path


def read_split(
    task: int, block: str, lang: str, frozen: bool = False
) -> pd.DataFrame:
    path = split_path(task, block, lang, frozen=frozen)
    if not path.exists():
        raise FileNotFoundError(f"no split at {path}")
    frame = pd.read_parquet(path)
    validate(frame, task)
    return frame


def read_block(task: int, block: str, frozen: bool = False) -> dict:
    """Every language split present for one block."""
    directory = split_path(task, block, "x", frozen=frozen).parent
    if not directory.exists():
        return {}
    return {
        path.stem: read_split(task, block, path.stem, frozen=frozen)
        for path in sorted(directory.glob("*.parquet"))
    }
