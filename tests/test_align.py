"""Tests for T-102b alignment."""

import pytest

from src.align import align_by_key, item_id_for


@pytest.fixture(scope="module")
def task3():
    return align_by_key(3)


def test_item_ids_are_deterministic_and_key_derived():
    assert item_id_for(3, "https://x/a") == item_id_for(3, "https://x/a")
    assert item_id_for(3, "https://x/a") != item_id_for(3, "https://x/b")
    # Different tasks never collide even on an identical key.
    assert item_id_for(2, "https://x/a") != item_id_for(3, "https://x/a")


def test_task3_aligns_completely(task3):
    assert task3.failures() == []
    assert task3.aligned_items == 532
    assert len(task3.frame) == 532 * 3


def test_every_item_appears_once_per_language(task3):
    counts = task3.frame.groupby("item_id")["lang"].nunique()
    assert set(counts) == {3}
    assert not task3.frame.duplicated(["item_id", "lang"]).any()


def test_aligned_items_share_one_label(task3):
    assert task3.label_disagreements == []
    assert (task3.frame.groupby("item_id")["label"].nunique() == 1).all()


def test_aligned_texts_differ_between_languages(task3):
    """Same item, genuinely different scripts — not an accidental self-join."""
    sample = task3.frame[task3.frame.item_id == task3.frame.item_id.iloc[0]]
    assert sample["text"].nunique() == 3


def test_independent_task_is_refused():
    """Task 1's languages hold different content; aligning it would invent one."""
    with pytest.raises(ValueError, match="no join key"):
        align_by_key(1)
    with pytest.raises(ValueError, match="no join key"):
        align_by_key(2)


def test_task2_has_no_exact_key():
    """Task 2 must go through embedding alignment, not a join."""
    with pytest.raises(ValueError, match="no join key"):
        align_by_key(2)


def test_task3_rejects_embedding_alignment():
    """Task 3 has an exact key; guessing at it with a model would be worse."""
    from src.align import align_by_embedding

    with pytest.raises(ValueError, match="exact join key"):
        align_by_embedding(3)
