"""Dataset loader for the training pipeline (CLAUDE2.md T-202).

Loads one frozen split by ``(block, lang, origin)``, tokenises it for a chosen
encoder, and hands the training loop a plain torch ``Dataset``. Every read goes
through :mod:`src.corpus_io`, so the §6 schema and the ``labels.json`` checks
(§4 rule 6) are enforced before a row is ever seen here.

Design notes:

* **Origin is an assertion, not a hint.** Each frozen file is single-origin, so
  asking for ``origin="native"`` on a machine-translated split is a mistake the
  loader must catch — training on MT would silently violate `hard rule 1`.
* **Phase 2 is classification.** Task 1 marks a numeral span and has no label,
  so this module refuses it rather than inventing a target.
* Module import stays torch-free; ``torch`` is imported lazily inside the
  dataset so the pure data helpers and their tests load fast.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .corpus_io import is_numeral_task, load_labels
from .corpus_io import read_split as _read_split
from .ids import BLOCK_NATIVE_LANG, block_for_native_lang
from .unicode_ranges import normalise_lang

# --------------------------------------------------------------------------
# Encoders
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Encoder:
    """An encoder Phase 2 can fine-tune. ``hf_id`` is the tokeniser source."""

    key: str
    hf_id: str


# CLAUDE2.md "Encoders". IndicBERT-v2 is all Phase 2 needs working; the other
# two are kept wired so the Phase 5 extension costs nothing.
ENCODERS: dict[str, Encoder] = {
    "indicbert-v2": Encoder("indicbert-v2", "ai4bharat/indic-bert"),
    "xlm-r-base": Encoder("xlm-r-base", "xlm-roberta-base"),
    "mbert-base": Encoder("mbert-base", "bert-base-multilingual-cased"),
}


def resolve_encoder(encoder: str | Encoder) -> Encoder:
    if isinstance(encoder, Encoder):
        return encoder
    try:
        return ENCODERS[encoder]
    except KeyError:
        raise ValueError(
            f"unknown encoder {encoder!r}; known: {sorted(ENCODERS)}"
        ) from None


# --------------------------------------------------------------------------
# Label metadata
# --------------------------------------------------------------------------


def label_names(task: int) -> list[str]:
    """Canonical class names for a task, in ``label_id`` order."""
    schema = load_labels(task)
    if schema is None:
        raise ValueError(
            f"task {task} has no labels in configs/labels.json — it is the "
            "numeral-span task and cannot be a classification target (T-202)"
        )
    return list(schema["labels"])


def num_labels(task: int) -> int:
    return len(label_names(task))


# --------------------------------------------------------------------------
# Split loading
# --------------------------------------------------------------------------


def load_split(
    task: int,
    block: str,
    lang: str,
    origin: str,
    *,
    frozen: bool = True,
) -> pd.DataFrame:
    """One split's rows for a given origin, validated and index-reset.

    ``origin`` is checked against the file's actual contents: a mismatch raises
    rather than returning an empty frame, so ``load_split(2, "H", "ben",
    "native")`` — Bengali in block H is machine-translated — fails loudly
    instead of handing back nothing.
    """
    if is_numeral_task(task):
        raise ValueError(
            f"task {task} is the numeral-span task and has no classification "
            "label; Phase 2 does not train on it (T-202)"
        )
    if origin not in ("native", "mt"):
        raise ValueError(f"origin must be 'native' or 'mt', got {origin!r}")

    frame = _read_split(task, block, normalise_lang(lang), frozen=frozen)
    matched = frame[frame["origin"] == origin]
    if matched.empty:
        present = sorted(frame["origin"].unique())
        raise ValueError(
            f"task {task} {block}/{lang} contains origin(s) {present}, "
            f"not {origin!r} — check hard rule 1 before training on this"
        )

    ids = matched["label_id"]
    k = num_labels(task)
    if not ids.between(0, k - 1).all():
        bad = sorted(set(ids[~ids.between(0, k - 1)]))
        raise ValueError(f"label_id outside 0..{k - 1}: {bad}")

    return matched.reset_index(drop=True)


def load_native(task: int, lang: str, *, frozen: bool = True) -> pd.DataFrame:
    """The native (human-written) split for a language — the only thing `hard
    rule 1` permits as training data.

    Raises for Malayalam, which has no native split anywhere (§2).
    """
    code = normalise_lang(lang)
    block = block_for_native_lang(code)
    return load_split(task, block, code, "native", frozen=frozen)


def native_languages() -> tuple[str, ...]:
    """The three languages with a native split: ``hin``, ``ben``, ``tel``."""
    return tuple(BLOCK_NATIVE_LANG.values())


# --------------------------------------------------------------------------
# Deterministic train / dev / test partitioning
# --------------------------------------------------------------------------


def stratified_split(
    frame: pd.DataFrame,
    *,
    seed: int,
    dev: float = 0.1,
    test: float = 0.0,
) -> dict[str, pd.DataFrame]:
    """Split one split's rows into ``train`` / ``dev`` / (optional) ``test``.

    The frozen corpus carries no train/test marker, so Phase 2 makes its own,
    seeded (§4 rule 7) and stratified on ``label_id`` so a small dev set does
    not skew class balance. Two runs with the same seed return byte-identical
    partitions; ``test`` keys appear only when ``test > 0``.
    """
    from sklearn.model_selection import train_test_split

    if not 0.0 <= dev < 1.0 or not 0.0 <= test < 1.0 or dev + test >= 1.0:
        raise ValueError(f"invalid fractions dev={dev} test={test}")

    holdout = dev + test
    if holdout == 0.0:
        return {"train": frame.reset_index(drop=True)}

    train, rest = train_test_split(
        frame,
        test_size=holdout,
        random_state=seed,
        stratify=frame["label_id"],
    )
    parts = {"train": train.reset_index(drop=True)}
    if test == 0.0:
        parts["dev"] = rest.reset_index(drop=True)
        return parts
    if dev == 0.0:
        parts["test"] = rest.reset_index(drop=True)
        return parts

    dev_frame, test_frame = train_test_split(
        rest,
        test_size=test / holdout,
        random_state=seed,
        stratify=rest["label_id"],
    )
    parts["dev"] = dev_frame.reset_index(drop=True)
    parts["test"] = test_frame.reset_index(drop=True)
    return parts


# --------------------------------------------------------------------------
# Tokenisation
# --------------------------------------------------------------------------


def get_tokenizer(encoder: str | Encoder):
    """Load the encoder's tokeniser. Needs a model download on first use."""
    from transformers import AutoTokenizer

    enc = resolve_encoder(encoder)
    return AutoTokenizer.from_pretrained(enc.hf_id)


class SplitDataset:
    """A tokenised split as a torch ``Dataset``.

    Text is tokenised once at construction with truncation to ``max_len`` and
    **no** padding; batches are padded to their own longest row by
    :meth:`collate`, which keeps wasted compute off the 4 GB card (T-205).
    ``item_id`` is carried through so predictions can be tied back to rows.
    """

    def __init__(self, frame: pd.DataFrame, tokenizer, max_len: int = 128):
        import torch

        self._torch = torch
        encoded = tokenizer(
            list(frame["text"].astype(str)),
            truncation=True,
            max_length=max_len,
            padding=False,
        )
        self.input_ids: list[list[int]] = encoded["input_ids"]
        self.attention_mask: list[list[int]] = encoded["attention_mask"]
        self.labels: list[int] = [int(x) for x in frame["label_id"]]
        self.item_ids: list[str] = [str(x) for x in frame["item_id"]]
        self.pad_id: int = tokenizer.pad_token_id or 0

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, i: int) -> dict:
        return {
            "input_ids": self.input_ids[i],
            "attention_mask": self.attention_mask[i],
            "label": self.labels[i],
            "item_id": self.item_ids[i],
        }

    def collate(self, batch: list[dict]) -> dict:
        """Pad a batch to its longest row and stack into tensors."""
        torch = self._torch
        width = max(len(row["input_ids"]) for row in batch)
        input_ids, attention, labels, item_ids = [], [], [], []
        for row in batch:
            pad = width - len(row["input_ids"])
            input_ids.append(row["input_ids"] + [self.pad_id] * pad)
            attention.append(row["attention_mask"] + [0] * pad)
            labels.append(row["label"])
            item_ids.append(row["item_id"])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "item_ids": item_ids,
        }


def load_dataset(
    task: int,
    lang: str,
    encoder: str | Encoder,
    *,
    origin: str = "native",
    block: str | None = None,
    max_len: int = 128,
    frozen: bool = True,
) -> SplitDataset:
    """Convenience: load a split and tokenise it in one call.

    ``block`` defaults to the native block for ``lang``; pass it explicitly for
    a machine-translated split (e.g. Malayalam, which has no native block).
    """
    code = normalise_lang(lang)
    if block is None:
        if origin != "native":
            raise ValueError(
                "block is required for a non-native split (there is no native "
                f"block for an {origin!r} arm)"
            )
        block = block_for_native_lang(code)
    frame = load_split(task, block, code, origin, frozen=frozen)
    return SplitDataset(frame, get_tokenizer(encoder), max_len=max_len)
