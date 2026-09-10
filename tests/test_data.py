"""Tests for T-202's dataset loader.

The offline block covers everything that decides whether the right rows and
labels reach training: origin is asserted not guessed, task 1 is refused, the
label range is checked, and the train/dev/test partition is seeded and
stratified. The ``slow`` block tokenises every language with every encoder —
§10's "all 4 languages × 3 encoders".
"""

import pandas as pd
import pytest

from src import data as D

NATIVE = [(2, "hin"), (2, "ben"), (2, "tel"), (3, "hin"), (3, "ben"), (3, "tel")]


# --------------------------------------------------------------------------
# Offline — no model
# --------------------------------------------------------------------------


@pytest.mark.parametrize("task,lang", NATIVE)
def test_native_split_loads_with_the_right_origin(task, lang):
    frame = D.load_native(task, lang)
    assert len(frame) > 0
    assert set(frame["origin"]) == {"native"}
    assert frame["src_lang"].isna().all()
    assert frame.index.tolist() == list(range(len(frame)))


def test_origin_is_asserted_not_guessed():
    """Bengali in block H is machine-translated; asking for native must fail."""
    with pytest.raises(ValueError, match="not 'native'"):
        D.load_split(2, "H", "ben", "native")
    mt = D.load_split(2, "H", "ben", "mt")
    assert set(mt["origin"]) == {"mt"}


def test_the_numeral_task_is_refused():
    with pytest.raises(ValueError, match="numeral"):
        D.load_split(1, "H", "hin", "native")
    with pytest.raises(ValueError, match="numeral"):
        D.label_names(1)


def test_malayalam_has_no_native_split():
    with pytest.raises(ValueError, match="no block|no native"):
        D.load_native(2, "mal")


def test_label_metadata_matches_the_schema():
    assert D.num_labels(2) == 2
    assert D.num_labels(3) == 10
    assert D.label_names(2) == ["sustainable", "unsustainable"]
    assert D.label_names(3)[0] == "climate change"


@pytest.mark.parametrize("task,lang", NATIVE)
def test_every_label_id_is_in_range(task, lang):
    frame = D.load_native(task, lang)
    assert frame["label_id"].between(0, D.num_labels(task) - 1).all()


def test_unknown_encoder_is_rejected():
    with pytest.raises(ValueError, match="unknown encoder"):
        D.resolve_encoder("bert-large-cased")


# --------------------------------------------------------------------------
# Partitioning
# --------------------------------------------------------------------------


def test_split_is_deterministic_for_a_seed():
    frame = D.load_native(2, "hin")
    a = D.stratified_split(frame, seed=7, dev=0.1, test=0.1)
    b = D.stratified_split(frame, seed=7, dev=0.1, test=0.1)
    for key in ("train", "dev", "test"):
        assert a[key]["item_id"].tolist() == b[key]["item_id"].tolist()


def test_a_different_seed_moves_rows():
    frame = D.load_native(2, "hin")
    a = D.stratified_split(frame, seed=1, dev=0.2)
    b = D.stratified_split(frame, seed=2, dev=0.2)
    assert a["dev"]["item_id"].tolist() != b["dev"]["item_id"].tolist()


def test_partitions_are_disjoint_and_cover_every_row():
    frame = D.load_native(3, "ben")
    parts = D.stratified_split(frame, seed=3, dev=0.15, test=0.15)
    ids = [set(p["item_id"]) for p in parts.values()]
    assert set.union(*ids) == set(frame["item_id"])
    assert sum(len(s) for s in ids) == len(frame)


def test_stratification_preserves_class_balance():
    frame = D.load_native(2, "tel")
    base = frame["label_id"].value_counts(normalize=True).sort_index()
    parts = D.stratified_split(frame, seed=5, dev=0.2)
    dev = parts["dev"]["label_id"].value_counts(normalize=True).sort_index()
    assert (base - dev).abs().max() < 0.03


def test_no_holdout_returns_train_only():
    frame = D.load_native(3, "hin")
    parts = D.stratified_split(frame, seed=1, dev=0.0, test=0.0)
    assert list(parts) == ["train"]
    assert len(parts["train"]) == len(frame)


def test_impossible_fractions_raise():
    frame = D.load_native(3, "hin")
    with pytest.raises(ValueError):
        D.stratified_split(frame, seed=1, dev=0.7, test=0.7)


# --------------------------------------------------------------------------
# slow — real tokenisers, all 4 languages × 3 encoders
# --------------------------------------------------------------------------

ENCODERS = ["indicbert-v2", "xlm-r-base", "mbert-base"]
LANG_SPLITS = [
    ("hin", "H", "native"),
    ("ben", "B", "native"),
    ("tel", "T", "native"),
    ("mal", "H", "mt"),
]


def _tokenizer_or_skip(encoder):
    """Load a tokeniser, skipping (not failing) if its repo is gated.

    ``ai4bharat/indic-bert`` is gated like the IndicTrans2 repos (CLAUDE.md
    §3.1): accept the terms at huggingface.co/ai4bharat/indic-bert and
    authenticate, then this runs.
    """
    try:
        return D.get_tokenizer(encoder)
    except OSError as exc:
        if "gated" in str(exc).lower() or "401" in str(exc):
            pytest.skip(f"{encoder}: repo not accessible ({exc})")
        raise


@pytest.mark.slow
@pytest.mark.parametrize("encoder", ENCODERS)
@pytest.mark.parametrize("lang,block,origin", LANG_SPLITS)
def test_split_tokenises_for_every_encoder(encoder, lang, block, origin):
    frame = D.load_split(2, block, lang, origin).head(24)
    tokenizer = _tokenizer_or_skip(encoder)
    ds = D.SplitDataset(frame, tokenizer, max_len=64)

    assert len(ds) == len(frame)
    for i in range(len(ds)):
        row = ds[i]
        assert 0 < len(row["input_ids"]) <= 64
        assert len(row["input_ids"]) == len(row["attention_mask"])
        assert row["label"] in (0, 1)

    batch = ds.collate([ds[i] for i in range(8)])
    width = batch["input_ids"].shape[1]
    assert batch["input_ids"].shape == (8, width)
    assert batch["attention_mask"].shape == (8, width)
    assert batch["labels"].shape == (8,)
    assert len(batch["item_ids"]) == 8
    # padding positions are masked
    assert ((batch["input_ids"] == ds.pad_id) & (batch["attention_mask"] == 1)).sum() == 0


@pytest.mark.slow
def test_load_dataset_one_call_shortcut():
    ds = D.load_dataset(3, "hin", "mbert-base", max_len=48)
    assert len(ds) == 532
    assert all(0 <= label < 10 for label in ds.labels)
