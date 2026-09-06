"""Key construction, block mapping and join validation.

The corpus is addressed by `(block_id, item_id, lang)` (§6). This module owns
how those keys are built and how a join over them is checked; it holds no
file-format knowledge, which belongs to ``corpus_io``.

Two id regimes exist, decided by whether a task's native splits are parallel
(§2.2):

* **Aligned tasks (2, 3)** — every language holds the same items, so `item_id`
  is *global*: the Hindi and Bengali rows for one item share an id, and
  `block_id` records which native language a row's translation descended from
  rather than which content set it belongs to.
* **Independent task (1)** — each language holds different sentences, so
  `item_id` is only meaningful *within* a block, exactly as §6 originally
  described.
"""

from __future__ import annotations

import hashlib

import pandas as pd

from .unicode_ranges import normalise_lang

# Each block is named for the language its native split is written in.
BLOCK_NATIVE_LANG = {"H": "hin", "B": "ben", "T": "tel"}
NATIVE_LANG_BLOCK = {lang: block for block, lang in BLOCK_NATIVE_LANG.items()}

# Every corpus language, including the one with no native split.
CORPUS_LANGS = ("hin", "ben", "tel", "mal")

BLOCKS = tuple(BLOCK_NATIVE_LANG)


def block_for_native_lang(lang: str) -> str:
    """`ben` -> `B`. Raises for Malayalam, which has no native split."""
    code = normalise_lang(lang)
    if code not in NATIVE_LANG_BLOCK:
        raise ValueError(
            f"{code!r} has no native split in IndicFinNLP and therefore no block; "
            f"native languages are {sorted(NATIVE_LANG_BLOCK)}"
        )
    return NATIVE_LANG_BLOCK[code]


def targets_for_block(block: str) -> tuple[str, ...]:
    """The three languages a block is translated into."""
    native = BLOCK_NATIVE_LANG[block]
    return tuple(lang for lang in CORPUS_LANGS if lang != native)


def local_item_id(
    task: int, block: str, text: str, discriminator: str | None = None
) -> str:
    """Id for a row that is not part of any cross-language parallel set.

    Derived from the content rather than the row number so it survives upstream
    reordering, and namespaced by block so two languages cannot collide.

    `discriminator` distinguishes rows that share a text but are different
    items. Task 1 needs it: a sentence containing three numbers appears three
    times, once per number, so the item is a *(sentence, span)* pair — 6776
    Hindi rows share a sentence with another row. Passing the span makes those
    ids distinct while keeping them content-derived.
    """
    payload = text if discriminator is None else f"{text}\u241f{discriminator}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"t{task}_{block}_{digest}"


def disambiguate(ids: list[str]) -> list[str]:
    """Append an occurrence suffix to genuinely repeated ids.

    Some rows are exact duplicates — same text, same label — and a
    content-derived id cannot separate them. Rule 1 forbids dropping them, so
    they are kept and numbered in upstream order instead. Ids that occur once
    are returned untouched, so the suffix never appears on normal rows.
    """
    from collections import Counter  # noqa: PLC0415

    totals = Counter(ids)
    seen: Counter = Counter()
    out = []
    for value in ids:
        if totals[value] == 1:
            out.append(value)
            continue
        seen[value] += 1
        out.append(f"{value}#{seen[value]}")
    return out


def validate_keys(frame: pd.DataFrame) -> None:
    """The primary key must actually be one."""
    duplicated = frame.duplicated(["block_id", "item_id", "lang"])
    if duplicated.any():
        offenders = frame.loc[duplicated, ["block_id", "item_id", "lang"]]
        raise ValueError(
            f"{int(duplicated.sum())} duplicate (block_id, item_id, lang) keys, "
            f"e.g.\n{offenders.head().to_string(index=False)}"
        )


def join_languages(frames: dict[str, pd.DataFrame], on: str = "item_id") -> pd.DataFrame:
    """Join one block's language splits into a row per item.

    Returns an inner join, so the row count *is* the answer to "how many items
    exist in every one of these languages". §8 requires that to equal N with no
    nulls before the corpus can be trusted.
    """
    if not frames:
        raise ValueError("nothing to join")
    joined = None
    for lang, frame in sorted(frames.items()):
        columns = {c: f"{c}_{lang}" for c in frame.columns if c != on}
        renamed = frame.rename(columns=columns)
        joined = renamed if joined is None else joined.merge(renamed, on=on, how="inner")
    return joined


def join_report(frames: dict[str, pd.DataFrame], on: str = "item_id") -> dict:
    """Diagnose a join instead of just counting it (§11).

    A bare row count says a join succeeded or failed; this says *which* items
    are missing from *which* language, which is what you need to fix it.
    """
    ids = {lang: set(frame[on]) for lang, frame in frames.items()}
    shared = set.intersection(*ids.values()) if ids else set()
    joined = join_languages(frames, on=on)
    return {
        "languages": sorted(frames),
        "per_language": {lang: len(values) for lang, values in ids.items()},
        "shared": len(shared),
        "joined_rows": len(joined),
        "nulls": int(joined.isna().sum().sum()),
        "missing_from": {
            lang: sorted(shared_all - values)[:5]
            for lang, values in ids.items()
            if (shared_all := set.union(*ids.values())) and (shared_all - values)
        },
    }
