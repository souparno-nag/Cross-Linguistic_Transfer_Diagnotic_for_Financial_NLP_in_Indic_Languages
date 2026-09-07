"""Turning a native split plus its translations into §6 corpus rows (T-106).

This sits between :mod:`translate`, which knows how to run the model and
nothing about the corpus, and :mod:`corpus_io`, which knows the schema and
nothing about translation. What lives here is the schema-aware half of
generation: which rows go to the model, how a translation becomes a row, and
what has to be true of the result before it is written.

Two things it is careful about:

* **Every source row produces exactly one MT row**, in the source's order, with
  the source's `item_id`. Row loss here is the unrecoverable corruption §1
  warns about, so the parity check is part of the module rather than left to
  the caller.
* **Identical sentences are translated once.** Task 1's item is a
  *(sentence, span)* pair, so a sentence containing three numbers is three
  rows sharing one text — 10640 Hindi rows hold 6357 distinct sentences.
  Translating each row separately would pay for the same sentence up to nine
  times over and, worse, could return three different translations of it,
  leaving the corpus with three versions of one sentence differing only by
  which number was being annotated.
"""

from __future__ import annotations

import pandas as pd

from .corpus_io import is_numeral_task, schema_columns
from .spans import recover_span

# Flags on a native row that describe the *item*, not its text, and therefore
# belong on its translations too. `unaligned` says the item has no counterpart
# in the other languages (§4 rule 1 keeps it); that stays true after translation.
INHERITED_FLAGS = ("unaligned",)


def representatives(frame: pd.DataFrame) -> pd.DataFrame:
    """One row per distinct text, in first-appearance order.

    First appearance rather than an arbitrary pick, so a resumed run selects
    the same representatives and reuses the checkpoint instead of retranslating.
    """
    return frame.drop_duplicates(subset="text", keep="first")[["item_id", "text"]]


def broadcast(frame: pd.DataFrame, translated: pd.DataFrame) -> pd.DataFrame:
    """Spread the representatives' translations back over every source row.

    `translated` is the output of :func:`translate.translate_rows` on
    :func:`representatives`. Returns one row per row of `frame`, in `frame`'s
    order.
    """
    lookup = representatives(frame).merge(translated, on="item_id", how="left").set_index("text")
    if lookup["translation"].isna().any():
        missing = int(lookup["translation"].isna().sum())
        raise ValueError(f"{missing} representative texts came back without a translation")
    out = pd.DataFrame(
        {
            "item_id": frame["item_id"].to_numpy(),
            "translation": frame["text"].map(lookup["translation"]).to_numpy(),
            "flags": frame["text"].map(lookup["flags"]).to_numpy(),
        }
    )
    return out


def build_mt_frame(
    task: int,
    source: pd.DataFrame,
    translations: pd.DataFrame,
    tgt_lang: str,
) -> pd.DataFrame:
    """One MT split: the source split's rows, rewritten in `tgt_lang`.

    `translations` carries `item_id`, `translation` and `flags`, one row per
    source row in the source's order — what :func:`broadcast` returns.

    Labels are copied across untouched (§4 rule 6). Task 1's offsets are not
    copied: they index the source string and mean nothing in a different
    script, so the numeral is re-located in the output and the outcome recorded
    in `span_recovered` (§6.1). A row whose number could not be found is
    flagged and kept, never dropped (rule 1).
    """
    if len(translations) != len(source):
        raise ValueError(
            f"{len(translations)} translations for {len(source)} source rows; "
            "row loss here corrupts every downstream result"
        )
    if list(translations["item_id"]) != list(source["item_id"]):
        raise ValueError("translations are not in the source's row order")

    src_lang = source["lang"].iloc[0]
    records = []
    for (_, row), (_, got) in zip(source.iterrows(), translations.iterrows()):
        text = str(got["translation"])
        flags = [f for f in row["flags"] if f in INHERITED_FLAGS]
        flags += [f for f in got["flags"] if f not in flags]

        record = {
            "block_id": row["block_id"],
            "item_id": row["item_id"],
            "lang": tgt_lang,
            "origin": "mt",
            "src_lang": src_lang,
            "text": text,
            "labse_sim": None,
        }
        if is_numeral_task(task):
            found = recover_span(text, str(row["number_english"]), str(row["text"]))
            record.update(
                {
                    "number_indic": found.matched,
                    "number_english": row["number_english"],
                    "start_posn": found.start,
                    "end_posn": found.end,
                    "magnitude": row["magnitude"],
                    "span_recovered": bool(found.recovered),
                }
            )
            flags += [f for f in found.flags if f not in flags]
        else:
            record.update({"label": row["label"], "label_id": row["label_id"]})
        record["flags"] = flags
        records.append(record)

    return pd.DataFrame(records, columns=schema_columns(task))


def check_split(task: int, source: pd.DataFrame, mt: pd.DataFrame) -> dict:
    """The T-106 acceptance criteria, as numbers rather than an eyeball.

    Raises on the two that invalidate the corpus — a row count that does not
    match the source, or an item that changed identity. Empty output and
    unrecovered spans are *reported*, not raised: they are findings the project
    exists to measure, and rule 1 keeps their rows.
    """
    if len(mt) != len(source):
        raise ValueError(f"row-count parity failed: {len(mt)} MT rows for {len(source)} source rows")
    if set(mt["item_id"]) != set(source["item_id"]):
        raise ValueError("MT item_ids do not match the source split's")

    empty = int((mt["text"].str.strip() == "").sum())
    unflagged = int(
        ((mt["text"].str.strip() == "") & ~mt["flags"].apply(lambda f: "empty_output" in f)).sum()
    )
    if unflagged:
        raise ValueError(f"{unflagged} empty translations are not flagged (§4 rule 1)")

    summary = {
        "rows": len(mt),
        "unique_source_texts": int(source["text"].nunique()),
        "empty": empty,
    }
    if is_numeral_task(task):
        if mt["span_recovered"].isna().any():
            raise ValueError("span_recovered must be populated on every task-1 MT row")
        summary["span_recovered"] = int(mt["span_recovered"].sum())
        summary["span_recovery_rate"] = round(float(mt["span_recovered"].mean()), 4)
        summary["span_scale_shift"] = int(
            mt["flags"].apply(lambda f: "span_scale_shift" in f).sum()
        )
        summary["span_ambiguous"] = int(
            mt["flags"].apply(lambda f: "span_ambiguous" in f).sum()
        )
    return summary
