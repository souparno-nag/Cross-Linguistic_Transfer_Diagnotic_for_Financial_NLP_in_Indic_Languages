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
    pairwise: dict[str, int] = field(default_factory=dict)
    partial_items: dict[str, int] = field(default_factory=dict)

    def failures(self) -> list[str]:
        """Conditions that invalidate the alignment.

        Unmatched rows are *not* a failure. Where the split sizes differ,
        some items genuinely exist in one language and not another; §4 rule 1
        keeps them and the report states how many. A conflicting label is a
        failure, because the corpus cannot then carry one gold label.
        """
        problems = []
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


def align_by_embedding(
    task: int,
    model_name: str = "sentence-transformers/LaBSE",
    tau: float = 0.82,
    batch_size: int = 64,
    device: str | None = None,
    cache_dir=None,
) -> AlignmentResult:
    """Align a task with no join key, by mutual nearest neighbour on embeddings.

    Three properties make this trustworthy rather than a guess:

    * **Mutual** — i pairs with j only when each is the other's single best
      match. A one-directional "closest match" would happily map ten different
      Hindi sentences onto one Bengali sentence.
    * **Above τ** — a mutual best match can still be a poor one. The threshold
      is what T-102's controls calibrated: genuinely unrelated same-domain
      content peaks around 0.5, real translations sit near 0.9.
    * **Three-way consistent** — hin↔ben, hin↔tel and ben↔tel must all agree on
      the same triple. A pair surviving two independent routes is far stronger
      evidence than one.

    No language is privileged: the three pairings are computed symmetrically and
    only triples on which all three agree are kept.
    """
    import numpy as np  # noqa: PLC0415

    from .audit import embed_all  # noqa: PLC0415

    spec = TASK_COLUMNS[task]
    if spec.get("join_key"):
        raise ValueError(
            f"task {task} has an exact join key; use align_by_key instead"
        )
    text_column, label_column = spec["text"], spec["label"]

    frames = {lang: load_split(task, lang) for lang in LANGUAGES}
    texts = {lang: frames[lang][text_column].tolist() for lang in LANGUAGES}
    embeddings = embed_all(texts, model_name, batch_size=batch_size, device=device)

    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        for lang, vectors in embeddings.items():
            np.save(cache_dir / f"{normalise_lang(lang)}.npy", vectors)

    def mutual(left: str, right: str) -> dict[int, int]:
        scores = embeddings[left] @ embeddings[right].T
        forward, reverse = scores.argmax(1), scores.argmax(0)
        return {
            i: int(j)
            for i, j in enumerate(forward)
            if reverse[j] == i and scores[i, j] >= tau
        }

    hin_ben = mutual("hindi", "bengali")
    hin_tel = mutual("hindi", "telugu")
    ben_tel = mutual("bengali", "telugu")
    pairwise = {
        "hindi-bengali": len(hin_ben),
        "hindi-telugu": len(hin_tel),
        "bengali-telugu": len(ben_tel),
    }

    triples = [
        (h, hin_ben[h], hin_tel[h])
        for h in hin_ben
        if h in hin_tel and ben_tel.get(hin_ben[h]) == hin_tel[h]
    ]

    rows = []
    for hindi_row, bengali_row, telugu_row in triples:
        positions = {"hindi": hindi_row, "bengali": bengali_row, "telugu": telugu_row}
        # Content-derived id, in fixed language order so it does not depend on
        # which language happened to be treated as the anchor.
        key = "\u241f".join(
            str(frames[lang].iloc[positions[lang]][text_column])
            for lang in LANGUAGES
        )
        item_id = item_id_for(task, key)
        for lang in LANGUAGES:
            record = frames[lang].iloc[positions[lang]]
            rows.append(
                {
                    "item_id": item_id,
                    "lang": normalise_lang(lang),
                    "source_row": int(positions[lang]),
                    "join_key": None,
                    "text": record[text_column],
                    "label": record[label_column],
                }
            )

    frame = pd.DataFrame(rows).sort_values(["item_id", "lang"]).reset_index(drop=True)

    disagreements = []
    if not frame.empty:
        for item_id, group in frame.groupby("item_id"):
            labels = set(group["label"])
            if len(labels) > 1:
                disagreements.append({"item_id": item_id, "labels": sorted(labels)})

    aligned_hindi = {h for h, _, _ in triples}
    aligned_bengali = {b for _, b, _ in triples}
    aligned_telugu = {t for _, _, t in triples}
    unmatched = {
        "hindi": [str(i) for i in range(len(texts["hindi"])) if i not in aligned_hindi],
        "bengali": [str(i) for i in range(len(texts["bengali"])) if i not in aligned_bengali],
        "telugu": [str(i) for i in range(len(texts["telugu"])) if i not in aligned_telugu],
    }

    return AlignmentResult(
        task=task,
        method=(
            f"mutual nearest neighbour on {model_name} at τ={tau}, "
            "three-way consistent"
        ),
        frame=frame,
        rows_per_language={lang: len(f) for lang, f in frames.items()},
        aligned_items=len(triples),
        unmatched=unmatched,
        label_disagreements=disagreements,
        duplicate_keys={},
        pairwise=pairwise,
        partial_items={
            "aligned_in_two_languages_only": len(hin_ben) + len(hin_tel) + len(ben_tel)
            - 3 * len(triples),
        },
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
    if result.pairwise:
        lines += [
            "Pairwise mutual matches before requiring three-way agreement:",
            "",
            "| Pair | Mutual matches above τ |",
            "|---|---|",
        ]
        lines += [f"| {pair} | {n} |" for pair, n in result.pairwise.items()]
        lines += [
            "",
            "Requiring all three pairings to agree on the same triple is what "
            f"reduces these to {result.aligned_items}. A pair surviving two "
            "independent routes is much stronger evidence than one.",
            "",
        ]
    dropped = {
        lang: len(keys) for lang, keys in result.unmatched.items() if keys
    }
    if dropped:
        lines += [
            "Rows that did not join the three-way set, by language: "
            + ", ".join(f"{lang} {n}" for lang, n in dropped.items())
            + ". These are **kept** in their own splits — §4 rule 1 forbids "
            "deleting them — but they cannot form a parallel item, either "
            "because the split sizes genuinely differ or because no confident "
            "mutual match was found.",
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
