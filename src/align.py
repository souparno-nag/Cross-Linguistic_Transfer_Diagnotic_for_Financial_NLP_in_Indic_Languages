"""Recover the item-to-item correspondence between native splits.

Tasks 2 and 3 are parallel — the same content in every language — but that
correspondence is not recorded anywhere in the released files. It has to be
reconstructed before the splits can be joined, because without it there is no
way to tell which items overlap between a language used for training and one
used for evaluation (CLAUDE.md §2.2).

Task 3 carries a `URL` column that identifies the underlying article, so its
alignment is an exact join and needs no model. Task 2 has no such key and needs
embedding search; that is a separate entry point, not yet written.

Task 1 is independently sourced and must not be aligned at all — its languages
hold genuinely different sentences.

Like ``audit.py`` this reads upstream spreadsheets directly and writes a
verification artefact rather than corpus data, so it predates the
``corpus_io.py`` rule in §5 rather than violating it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import pandas as pd

from .audit import TASK_COLUMNS, config_hash, load_split
from .unicode_ranges import normalise_lang

LANGUAGES = ("hindi", "bengali", "telugu")


def item_id_for(task: int, key: str) -> str:
    """Deterministic id for one aligned item.

    Derived from the join key rather than from row position, so the id survives
    upstream adding, removing or reordering rows — a positional id would
    silently re-point at a different article (§4 rule 7).
    """
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
    return f"t{task}_{digest}"


@dataclass
class AlignmentResult:
    """The recovered mapping plus everything needed to trust it."""

    task: int
    method: str
    frame: pd.DataFrame
    rows_per_language: dict[str, int] = field(default_factory=dict)
    aligned_items: int = 0
    unmatched: dict[str, list[str]] = field(default_factory=dict)
    label_disagreements: list[dict] = field(default_factory=list)
    duplicate_keys: dict[str, int] = field(default_factory=dict)

    def failures(self) -> list[str]:
        problems = []
        for lang, keys in self.unmatched.items():
            if keys:
                problems.append(
                    f"{lang}: {len(keys)} items missing from at least one other "
                    f"language, e.g. {keys[0]}"
                )
        if self.label_disagreements:
            first = self.label_disagreements[0]
            problems.append(
                f"{len(self.label_disagreements)} items carry different labels "
                f"across languages, e.g. {first['item_id']}: {first['labels']}"
            )
        for lang, count in self.duplicate_keys.items():
            if count:
                problems.append(f"{lang}: {count} duplicate join keys")
        return problems


def align_by_key(task: int) -> AlignmentResult:
    """Align a task's native splits on its shared join key.

    Only valid for a task whose upstream release carries a column identifying
    the underlying item across languages — currently task 3's ``URL``.
    """
    spec = TASK_COLUMNS[task]
    key_column = spec.get("join_key")
    if not key_column:
        raise ValueError(
            f"task {task} has no join key; it needs embedding alignment, or is "
            "independently sourced and must not be aligned at all"
        )
    text_column, label_column = spec["text"], spec["label"]

    frames = {lang: load_split(task, lang) for lang in LANGUAGES}
    keys = {lang: frames[lang][key_column].tolist() for lang in LANGUAGES}

    duplicates = {
        lang: len(values) - len(set(values)) for lang, values in keys.items()
    }
    shared = set.intersection(*(set(values) for values in keys.values()))
    unmatched = {
        lang: sorted(set(values) - shared) for lang, values in keys.items()
    }

    rows = []
    for lang in LANGUAGES:
        frame = frames[lang]
        for position, record in frame.iterrows():
            key = record[key_column]
            if key not in shared:
                continue
            rows.append(
                {
                    "item_id": item_id_for(task, key),
                    "lang": normalise_lang(lang),
                    "source_row": int(position),
                    "join_key": key,
                    "text": record[text_column],
                    "label": record[label_column],
                }
            )
    frame = pd.DataFrame(rows).sort_values(["item_id", "lang"]).reset_index(drop=True)

    # The same item must carry the same label in every language. If it does
    # not, one of the two is wrong and the corpus cannot carry a single gold
    # label for that item (§4 rule 6).
    disagreements = []
    for item_id, group in frame.groupby("item_id"):
        labels = set(group["label"])
        if len(labels) > 1:
            disagreements.append({"item_id": item_id, "labels": sorted(labels)})

    return AlignmentResult(
        task=task,
        method=f"exact join on `{key_column}`",
        frame=frame,
        rows_per_language={lang: len(f) for lang, f in frames.items()},
        aligned_items=len(shared),
        unmatched=unmatched,
        label_disagreements=disagreements,
        duplicate_keys=duplicates,
    )


def alignment_config(task: int, method: str) -> dict:
    return {"task": task, "method": method, "languages": list(LANGUAGES)}


def summarise(result: AlignmentResult) -> str:
    """Human-readable report body."""
    config = alignment_config(result.task, result.method)
    lines = [
        f"# T-102b — Task {result.task} alignment",
        "",
        f"Method: {result.method} · config hash `{config_hash(config)}`",
        "",
        "| Language | Upstream rows | Aligned rows |",
        "|---|---|---|",
    ]
    counts = result.frame.groupby("lang").size().to_dict()
    for lang in LANGUAGES:
        code = normalise_lang(lang)
        lines.append(
            f"| {lang} | {result.rows_per_language[lang]} | {counts.get(code, 0)} |"
        )
    lines += [
        "",
        f"**{result.aligned_items} items** align across all three languages.",
        "",
    ]
    dropped = {
        lang: len(keys) for lang, keys in result.unmatched.items() if keys
    }
    if dropped:
        lines += [
            "Items present in one language but not all three — kept in their own "
            "split, but not part of the parallel set (§4 rule 1 forbids deleting "
            f"them): {dropped}",
            "",
        ]
    else:
        lines += [
            "Every upstream row aligned; no item is present in one language and "
            "missing from another.",
            "",
        ]
    lines += [
        "Label agreement across languages: "
        + (
            f"**{len(result.label_disagreements)} disagreements**"
            if result.label_disagreements
            else "**complete** — every aligned item carries the same label in all "
            "three languages."
        ),
        "",
    ]
    return "\n".join(lines)
