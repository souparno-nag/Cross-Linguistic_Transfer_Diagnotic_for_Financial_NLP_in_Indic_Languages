"""T-103 — build the label schema and ingest the native splits.

    python -m scripts.t103_ingest --task 2

Reads the upstream spreadsheets once, at this boundary, and writes the three
native splits as Parquet under `data/raw/task_{n}/{block}/{lang}.parquet`.
Nothing downstream touches `.xlsx` again (§5).

Where a task's languages are parallel, item ids come from the alignment map
built in T-102b so the same item carries one id in every language. Rows that
did not align are still ingested — §4 rule 1 forbids dropping them — with a
locally-scoped id and an `unaligned` flag, so they stay available while being
clearly outside the parallel set.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from src.align import LANGUAGES
from src.audit import TASK_COLUMNS, config_hash, load_split
from src.corpus_io import (
    LABELS_PATH,
    is_numeral_task,
    load_labels,
    schema_columns,
    split_path,
    write_labels,
    write_split,
)
from src.download_dataset.paths import REPO_ROOT
from src.ids import block_for_native_lang, disambiguate, local_item_id
from src.unicode_ranges import normalise_lang

ALIGNMENT_ROOT = REPO_ROOT / "data" / "verification"
CLASSIFICATION_TASKS = (2, 3)


def build_label_schema() -> dict:
    """Reconcile labels across all languages of every classification task.

    Sorted so the canonical index is reproducible from the data alone. Once
    written this ordering is frozen — §4 rule 6 forbids reordering, because a
    `label_id` already baked into an artefact would silently change meaning.
    """
    schema = {}
    for task in CLASSIFICATION_TASKS:
        column = TASK_COLUMNS[task]["label"]
        per_language = {
            lang: sorted(set(load_split(task, lang)[column])) for lang in LANGUAGES
        }
        union = sorted(set().union(*per_language.values()))
        disagreeing = {
            lang: sorted(set(union) - set(values))
            for lang, values in per_language.items()
            if set(values) != set(union)
        }
        if disagreeing:
            raise ValueError(
                f"task {task}: languages disagree on the label set: {disagreeing}. "
                "Reconcile deliberately; do not let the union paper over it."
            )
        schema[f"task_{task}"] = {
            "column": column,
            "labels": union,
            "note": "index is position in this list; frozen once written (§4 rule 6)",
        }
    schema["task_1"] = None
    return schema


def load_alignment(task: int) -> pd.DataFrame | None:
    path = ALIGNMENT_ROOT / f"task_{task}" / "alignment.parquet"
    return pd.read_parquet(path) if path.exists() else None


def build_native_split(task: int, lang: str, alignment: pd.DataFrame | None):
    """One native split, every upstream row kept."""
    code = normalise_lang(lang)
    block = block_for_native_lang(lang)
    spec = TASK_COLUMNS[task]
    source = load_split(task, lang)

    # Rows that aligned across languages take the shared id; the rest keep a
    # block-local one and are flagged rather than dropped.
    shared = {}
    if alignment is not None:
        rows = alignment[alignment["lang"] == code]
        shared = dict(zip(rows["source_row"], rows["item_id"]))

    records = []
    for position, record in source.iterrows():
        text = str(record[spec["text"]])
        item_id = shared.get(position)
        flags = []
        if item_id is None:
            # Task 1's item is a (sentence, number) pair, so the span is part
            # of its identity; other tasks identify an item by its text alone.
            discriminator = (
                f"{record['start_posn']}:{record['end_posn']}"
                if is_numeral_task(task)
                else None
            )
            item_id = local_item_id(task, block, text, discriminator)
            if alignment is not None:
                flags.append("unaligned")
        row = {
            "block_id": block,
            "item_id": item_id,
            "lang": code,
            "origin": "native",
            "src_lang": None,
            "text": text,
            "labse_sim": None,
            "flags": flags,
        }
        if is_numeral_task(task):
            row.update(
                {
                    "number_indic": str(record["number_indic"]),
                    "number_english": str(record["number_english"]),
                    "start_posn": int(record["start_posn"]),
                    "end_posn": int(record["end_posn"]),
                    "magnitude": str(record["magnitude"]),
                    "span_recovered": None,
                }
            )
        else:
            schema = load_labels(task)
            label = str(record[spec["label"]])
            row.update({"label": label, "label_id": schema["index"][label]})
        records.append(row)

    frame = pd.DataFrame(records, columns=schema_columns(task))
    # Exact-duplicate rows survive content-derived ids only with a suffix.
    frame["item_id"] = disambiguate(frame["item_id"].tolist())
    return frame, block


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=int, required=True)
    args = parser.parse_args(argv)

    if args.task not in TASK_COLUMNS:
        print(f"unknown task {args.task}", file=sys.stderr)
        return 2

    write_labels(build_label_schema())
    print(f"Wrote {LABELS_PATH.relative_to(REPO_ROOT)}")

    alignment = load_alignment(args.task)
    if alignment is None and args.task != 1:
        print(
            f"No alignment map for task {args.task}. Run "
            f"`python -m scripts.t102b_align --task {args.task}` first.",
            file=sys.stderr,
        )
        return 2

    config = {"task": args.task, "aligned": alignment is not None}
    totals = {}
    for lang in LANGUAGES:
        frame, block = build_native_split(args.task, lang, alignment)
        path = write_split(frame, args.task, block, normalise_lang(lang))
        unaligned = sum(1 for f in frame["flags"] if "unaligned" in f)
        totals[lang] = (len(frame), unaligned)
        detail = (
            f"({len(frame) - unaligned} parallel, {unaligned} unaligned) "
            if alignment is not None
            else "(independently sourced; no alignment applies) "
        )
        print(
            f"  {block}/{normalise_lang(lang)}: {len(frame)} rows {detail}"
            f"-> {path.relative_to(REPO_ROOT)}"
        )

    print(f"\nconfig {config_hash(config)}")
    print(f"{sum(n for n, _ in totals.values())} native rows ingested, none dropped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
