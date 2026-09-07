"""Tests for T-106's split construction.

No model is loaded. What is covered is the part that decides whether the corpus
is trustworthy: that every source row produces exactly one MT row with the
same id and label, that spans are re-derived rather than carried across, and
that nothing is ever dropped (§4 rule 1).
"""

import pandas as pd
import pytest

from src import generate as G
from src.corpus_io import schema_columns, validate


def native_classification(n=3):
    return pd.DataFrame(
        {
            "block_id": ["H"] * n,
            "item_id": [f"t2_H_{i}" for i in range(n)],
            "lang": ["hin"] * n,
            "origin": ["native"] * n,
            "src_lang": [None] * n,
            "text": [f"वाक्य {i}" for i in range(n)],
            "label": ["sustainable", "unsustainable", "sustainable"][:n],
            "label_id": [0, 1, 0][:n],
            "labse_sim": [None] * n,
            "flags": [[], ["unaligned"], []][:n],
        },
        columns=schema_columns(2),
    )


def native_numeral():
    """Two rows sharing one sentence — task 1's (sentence, span) item."""
    text = "एनपीसीआई द्वारा 24 घंटे और 7 दिन"
    return pd.DataFrame(
        {
            "block_id": ["H", "H"],
            "item_id": ["t1_H_a", "t1_H_b"],
            "lang": ["hin", "hin"],
            "origin": ["native", "native"],
            "src_lang": [None, None],
            "text": [text, text],
            "number_indic": ["24", "7"],
            "number_english": ["24", "7"],
            "start_posn": [text.index("24"), text.index("7")],
            "end_posn": [text.index("24") + 2, text.index("7") + 1],
            "magnitude": ["1", "0"],
            "span_recovered": [None, None],
            "labse_sim": [None, None],
            "flags": [[], []],
        },
        columns=schema_columns(1),
    )


def translated(frame, rendering):
    return pd.DataFrame(
        {
            "item_id": frame["item_id"],
            "translation": [rendering(t) for t in frame["text"]],
            "flags": [[] for _ in range(len(frame))],
        }
    ).reset_index(drop=True)


def test_identical_sentences_are_translated_once():
    native = native_numeral()
    assert len(G.representatives(native)) == 1, "one sentence, two annotated numbers"


def test_broadcast_gives_every_row_the_shared_translation():
    native = native_numeral()
    reps = G.representatives(native)
    out = G.broadcast(native, translated(reps, lambda t: "[ben] " + t))
    assert list(out["item_id"]) == list(native["item_id"])
    assert out["translation"].nunique() == 1, "one sentence must not become two"


def test_labels_and_ids_are_carried_through_untouched():
    native = native_classification()
    mt = G.build_mt_frame(2, native, translated(native, lambda t: "[ben] " + t), "ben")
    validate(mt, 2)
    assert list(mt["item_id"]) == list(native["item_id"])
    assert list(mt["label"]) == list(native["label"])
    assert list(mt["label_id"]) == list(native["label_id"])
    assert set(mt["origin"]) == {"mt"} and set(mt["src_lang"]) == {"hin"}
    assert mt["labse_sim"].isna().all()


def test_unaligned_is_inherited_by_the_translation():
    native = native_classification()
    mt = G.build_mt_frame(2, native, translated(native, lambda t: t), "ben")
    assert "unaligned" in mt["flags"].iloc[1]
    assert mt["flags"].iloc[0] == []


def test_spans_are_recovered_not_carried_across():
    """The offsets must index the *translation*, not the source string."""
    native = native_numeral()
    reps = G.representatives(native)
    out = G.broadcast(native, translated(reps, lambda t: "NPCI-യുടെ 24 മണിക്കൂറും 7 ദിവസവും"))
    mt = G.build_mt_frame(1, native, out, "mal")
    validate(mt, 1)
    assert list(mt["span_recovered"]) == [True, True]
    for _, row in mt.iterrows():
        assert row["text"][row["start_posn"] : row["end_posn"]] == row["number_indic"]
    assert list(mt["start_posn"]) != list(native["start_posn"])


def test_a_lost_number_is_flagged_and_kept():
    native = native_numeral()
    reps = G.representatives(native)
    out = G.broadcast(native, translated(reps, lambda t: "NPCI സംവിധാനം"))
    mt = G.build_mt_frame(1, native, out, "mal")
    assert len(mt) == len(native), "rule 1: a failed recovery never drops a row"
    assert list(mt["span_recovered"]) == [False, False]
    assert all("span_not_recovered" in f for f in mt["flags"])
    assert list(mt["start_posn"]) == [-1, -1]


def test_empty_translation_is_kept_and_stays_flagged():
    native = native_classification()
    out = pd.DataFrame(
        {
            "item_id": native["item_id"],
            "translation": ["", "x", "y"],
            "flags": [["empty_output"], [], []],
        }
    )
    mt = G.build_mt_frame(2, native, out, "ben")
    validate(mt, 2)
    summary = G.check_split(2, native, mt)
    assert summary["rows"] == len(native) and summary["empty"] == 1


def test_an_unflagged_empty_translation_is_a_failure():
    native = native_classification()
    out = pd.DataFrame(
        {"item_id": native["item_id"], "translation": ["", "x", "y"], "flags": [[], [], []]}
    )
    mt = G.build_mt_frame(2, native, out, "ben")
    with pytest.raises(ValueError, match="not flagged"):
        G.check_split(2, native, mt)


def test_row_loss_is_refused_rather_than_absorbed():
    native = native_classification()
    short = translated(native, lambda t: t).head(2)
    with pytest.raises(ValueError, match="row loss"):
        G.build_mt_frame(2, native, short, "ben")


def test_reordered_translations_are_refused():
    native = native_classification()
    shuffled = translated(native, lambda t: t).iloc[::-1].reset_index(drop=True)
    with pytest.raises(ValueError, match="row order"):
        G.build_mt_frame(2, native, shuffled, "ben")


def test_check_split_reports_span_recovery_rate():
    native = native_numeral()
    reps = G.representatives(native)
    out = G.broadcast(native, translated(reps, lambda t: "24 മണിക്കൂർ"))
    mt = G.build_mt_frame(1, native, out, "mal")
    summary = G.check_split(1, native, mt)
    assert summary["span_recovered"] == 1 and summary["span_recovery_rate"] == 0.5


def test_a_partial_run_does_not_wipe_the_rest_of_the_report(tmp_path):
    """`--block H --targets tel` must not leave a 9-direction task showing one.

    It did: regenerating one direction of task 2 overwrote the report with its
    single row, losing the other eight.
    """
    import json

    from scripts.t106_generate import merge_directions

    config = {"decoding_fingerprint": "abc", "device": "cuda"}
    path = tmp_path / "t106_generate.json"
    path.write_text(
        json.dumps(
            {
                "config": config,
                "directions": [
                    {"block": "H", "src_lang": "hin", "tgt_lang": "tel", "rows": 1},
                    {"block": "B", "src_lang": "ben", "tgt_lang": "hin", "rows": 2},
                ],
                "failed": [],
            }
        )
    )
    rerun = [{"block": "H", "src_lang": "hin", "tgt_lang": "tel", "rows": 99}]
    merged, failed = merge_directions(path, rerun, config, [])
    assert len(merged) == 2, "the untouched direction must survive"
    redone = next(d for d in merged if d["tgt_lang"] == "tel")
    assert redone["rows"] == 99, "the rerun direction must be the new one"


def test_a_different_fingerprint_replaces_rather_than_merges(tmp_path):
    """Output from another model or decoding config is not comparable (§8)."""
    import json

    from scripts.t106_generate import merge_directions

    path = tmp_path / "t106_generate.json"
    path.write_text(
        json.dumps(
            {
                "config": {"decoding_fingerprint": "old", "device": "cuda"},
                "directions": [{"block": "B", "src_lang": "ben", "tgt_lang": "hin", "rows": 2}],
                "failed": [],
            }
        )
    )
    rerun = [{"block": "H", "src_lang": "hin", "tgt_lang": "tel", "rows": 1}]
    merged, _ = merge_directions(path, rerun, {"decoding_fingerprint": "new", "device": "cuda"}, [])
    assert merged == rerun
