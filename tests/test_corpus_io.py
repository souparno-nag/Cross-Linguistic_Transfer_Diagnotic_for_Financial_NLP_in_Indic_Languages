"""Tests for the T-103 IO layer, ID scheme and label schema."""

import pandas as pd
import pytest

from src.corpus_io import (
    load_labels,
    read_split,
    schema_columns,
    split_path,
    validate,
    write_split,
)
from src.ids import (
    block_for_native_lang,
    disambiguate,
    join_languages,
    join_report,
    local_item_id,
    validate_keys,
)

# §10 requires a round-trip fixture covering all four scripts, Telugu digits,
# a currency symbol and a lakh/crore expression.
FIXTURE_TEXTS = [
    "കാർബൺ ഫുട്പ്രിന്റ് കുറയ്ക്കുന്നു",          # Malayalam
    "కంపెనీ ౧౨౩౪ కోట్ల రూపాయలు పెట్టుబడి పెట్టింది",  # Telugu digits
    "कंपनी ने ₹2.5 लाख का निवेश किया",              # currency + lakh
    "কোম্পানি ১০ কোটি টাকা বিনিয়োগ করেছে।",         # Bengali + crore + danda
]


def _frame(task=2, n=4):
    schema = load_labels(task)
    labels = schema["labels"]
    return pd.DataFrame(
        [
            {
                "block_id": "H",
                "item_id": f"t{task}_H_fixture{i}",
                "lang": "mal",
                "origin": "mt",
                "src_lang": "hin",
                "text": FIXTURE_TEXTS[i % len(FIXTURE_TEXTS)],
                "label": labels[i % len(labels)],
                "label_id": i % len(labels),
                "labse_sim": 0.9,
                "flags": [],
            }
            for i in range(n)
        ],
        columns=schema_columns(task),
    )


def test_roundtrip_preserves_every_script_byte_for_byte(tmp_path, monkeypatch):
    import src.corpus_io as io

    monkeypatch.setattr(io, "CORPUS_ROOT", tmp_path)
    frame = _frame()
    write_split(frame, 2, "H", "mal")
    back = read_split(2, "H", "mal")
    assert back["text"].tolist() == frame["text"].tolist()
    for original, restored in zip(frame["text"], back["text"]):
        assert original.encode("utf-8") == restored.encode("utf-8")
    pd.testing.assert_frame_equal(frame, back)


def test_loader_raises_on_unknown_label():
    """Rule 6: labels are never coerced or skipped, only refused."""
    frame = _frame()
    frame.loc[0, "label"] = "definitely-not-a-real-label"
    with pytest.raises(ValueError, match="outside configs/labels.json"):
        validate(frame, 2)


def test_label_id_must_agree_with_schema():
    frame = _frame()
    frame.loc[0, "label_id"] = 99
    with pytest.raises(ValueError, match="label_id disagreeing"):
        validate(frame, 2)


def test_schema_order_is_enforced():
    frame = _frame()[list(reversed(schema_columns(2)))]
    with pytest.raises(ValueError, match="schema mismatch"):
        validate(frame, 2)


def test_native_rows_may_not_claim_a_source():
    frame = _frame()
    frame.loc[0, "origin"] = "native"
    with pytest.raises(ValueError, match="native rows must have a null src_lang"):
        validate(frame, 2)


def test_mt_rows_must_record_their_source():
    frame = _frame()
    frame.loc[0, "src_lang"] = None
    with pytest.raises(ValueError, match="mt rows must record src_lang"):
        validate(frame, 2)


def test_duplicate_keys_are_refused():
    frame = pd.concat([_frame(n=1), _frame(n=1)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        validate_keys(frame)


def test_disambiguate_only_touches_repeats():
    assert disambiguate(["a", "b", "c"]) == ["a", "b", "c"]
    assert disambiguate(["a", "a", "b"]) == ["a#1", "a#2", "b"]


def test_item_ids_distinguish_task1_spans():
    """A sentence with two numbers is two items, not one."""
    text = "कंपनी ने 10 और 20 करोड़ निवेश किए"
    assert local_item_id(1, "H", text, "9:11") != local_item_id(1, "H", text, "15:17")
    assert local_item_id(1, "H", text, "9:11") == local_item_id(1, "H", text, "9:11")


def test_malayalam_has_no_block():
    with pytest.raises(ValueError, match="no native split"):
        block_for_native_lang("malayalam")


# --- acceptance: joins over real ingested data -----------------------------


@pytest.mark.parametrize("task,expected", [(2, 1769), (3, 532)])
def test_aligned_tasks_join_across_languages(task, expected):
    """The parallel set joins to exactly the number T-102b aligned, no nulls."""
    frames = {
        lang: read_split(task, block, lang)
        for block, lang in (("H", "hin"), ("B", "ben"), ("T", "tel"))
    }
    parallel = {
        lang: f[~f["flags"].apply(lambda x: "unaligned" in list(x))]
        for lang, f in frames.items()
    }
    report = join_report(parallel)
    assert report["joined_rows"] == expected
    assert report["shared"] == expected
    joined = join_languages(parallel)
    assert joined["item_id"].is_unique
    assert not joined[[c for c in joined.columns if c.startswith("text_")]].isna().any().any()


def test_task1_languages_must_not_join():
    """Task 1 is independently sourced; a cross-language join is meaningless."""
    frames = {
        lang: read_split(1, block, lang)
        for block, lang in (("H", "hin"), ("B", "ben"), ("T", "tel"))
    }
    assert join_report(frames)["joined_rows"] == 0


def test_every_ingested_split_validates():
    for task in (1, 2, 3):
        for block, lang in (("H", "hin"), ("B", "ben"), ("T", "tel")):
            frame = read_split(task, block, lang)
            assert len(frame) > 0
            assert frame["origin"].eq("native").all()
            assert split_path(task, block, lang).exists()
