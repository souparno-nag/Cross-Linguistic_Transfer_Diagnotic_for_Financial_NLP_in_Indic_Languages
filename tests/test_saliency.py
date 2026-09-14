"""Tests for T-605 (this build's T-604) — the Integrated Gradients /
terminology-gap module.

`compute_salience` needs torch + Captum + a loaded checkpoint, so it is
exercised only by the `@pytest.mark.slow` test. Everything else — lexicon
lookup, token-span recovery, the divergence arithmetic, and
`diagnose_instance`'s fire / not_fired / not_applicable branching — is pure
Python and is tested here with `compute_salience` monkeypatched out, on
synthetic salience arrays.
"""

from __future__ import annotations

import pytest

from src import saliency as S


def _ws_offsets(text: str) -> list[tuple[int, int]]:
    """Char offsets for each whitespace-separated token in `text` — a
    trivial stand-in for a real tokenizer's `offset_mapping`."""
    offsets = []
    idx = 0
    for word in text.split(" "):
        start = text.index(word, idx)
        end = start + len(word)
        offsets.append((start, end))
        idx = end
    return offsets


# --------------------------------------------------------------------------
# find_lexicon_term
# --------------------------------------------------------------------------


def test_find_lexicon_term_matches_first_present_term():
    concepts = [{"id": "a", "hin": "नहीं"}, {"id": "b", "hin": "ऊर्जा"}]
    found = S.find_lexicon_term("ऊर्जा बढ़ी।", concepts, "hin")
    assert found == ("ऊर्जा", concepts[1])


def test_find_lexicon_term_skips_null_surface_forms():
    concepts = [{"id": "a", "hin": None}, {"id": "b", "hin": "ऊर्जा"}]
    found = S.find_lexicon_term("ऊर्जा बढ़ी।", concepts, "hin")
    assert found == ("ऊर्जा", concepts[1])


def test_find_lexicon_term_returns_none_when_absent():
    concepts = [{"id": "a", "hin": "जलवायु"}]
    assert S.find_lexicon_term("बाजार में वृद्धि हुई।", concepts, "hin") is None


# --------------------------------------------------------------------------
# term_token_span_from_offsets / salience_share / fires
# --------------------------------------------------------------------------


def test_term_token_span_locates_the_right_tokens():
    text = "the energy sector grew"
    offsets = _ws_offsets(text)
    span = S.term_token_span_from_offsets(text, "energy", offsets)
    assert span == [1]


def test_term_token_span_returns_none_when_term_absent():
    text = "the market grew"
    offsets = _ws_offsets(text)
    assert S.term_token_span_from_offsets(text, "energy", offsets) is None


def test_salience_share_is_a_fraction_of_the_total():
    assert S.salience_share([8.0, 2.0], [0]) == pytest.approx(0.8)


def test_salience_share_zero_total_is_zero_not_an_error():
    assert S.salience_share([0.0, 0.0], [0]) == 0.0


def test_fires_boundary_is_inclusive():
    assert S.fires(0.3) is True
    assert S.fires(0.2999) is False


# --------------------------------------------------------------------------
# diagnose_instance — 5 positive (fires) fixtures
# --------------------------------------------------------------------------


class _FakeCompute:
    """Replaces `compute_salience`: returns queued `(salience, offsets,
    delta)` results in call order (source first, then target)."""

    def __init__(self, responses):
        self.responses = list(responses)

    def __call__(self, loaded, tokenizer, text, gold_label, *, max_len, n_steps, device):
        return self.responses.pop(0)


POSITIVE_FIXTURES = [
    # (source, term, source_salience, target, tgt_term, target_salience) -- divergence >= 0.3
    ("energy sector grew", "energy", [8.0, 1.0, 1.0], "urja sector grew", "urja", [0.2, 4.8, 5.0]),
    ("carbon levels rose", "carbon", [9.0, 0.5, 0.5], "karbon levels rose", "karbon", [0.1, 4.9, 5.0]),
    ("waste management plan", "waste", [7.0, 1.5, 1.5], "west management plan", "west", [0.05, 4.9, 5.05]),
    ("the emission rate fell", "emission", [1.0, 8.0, 0.5, 0.5], "the emishun rate fell", "emishun", [1.0, 0.3, 4.35, 4.35]),
    ("pollution control act", "pollution", [6.0, 2.0, 2.0], "polushun control act", "polushun", [0.02, 4.99, 4.99]),
]


@pytest.mark.parametrize("source,term,source_salience,target,tgt_term,target_salience", POSITIVE_FIXTURES)
def test_positive_fixtures_fire(monkeypatch, source, term, source_salience, target, tgt_term, target_salience):
    concepts = [{"id": "concept", "hin": term, "ben": tgt_term}]
    fake = _FakeCompute([
        (source_salience, _ws_offsets(source), 0.01),
        (target_salience, _ws_offsets(target), 0.01),
    ])
    monkeypatch.setattr(S, "compute_salience", fake)

    result = S.diagnose_instance(
        loaded=None, tokenizer=None, source_text=source, target_text=target,
        gold_label=0, concepts=concepts, lang_src="hin", lang_tgt="ben", max_len=32,
    )
    assert result.status == "fired", f"expected fire: {source!r} -> {target!r}, divergence={result.divergence}"
    assert result.divergence >= S.DEFAULT_DIVERGENCE_THRESHOLD


# --------------------------------------------------------------------------
# diagnose_instance — 5 negative fixtures (not_fired / not_applicable)
# --------------------------------------------------------------------------


def test_not_applicable_when_source_has_no_lexicon_term(monkeypatch):
    monkeypatch.setattr(S, "compute_salience", _FakeCompute([]))
    result = S.diagnose_instance(
        loaded=None, tokenizer=None, source_text="market grew today", target_text="anything",
        gold_label=0, concepts=[{"id": "c", "hin": "ऊर्जा", "ben": "শক্তি"}],
        lang_src="hin", lang_tgt="ben", max_len=32,
    )
    assert result.status == "not_applicable"


def test_not_applicable_when_target_surface_form_missing(monkeypatch):
    monkeypatch.setattr(S, "compute_salience", _FakeCompute([]))
    result = S.diagnose_instance(
        loaded=None, tokenizer=None, source_text="energy sector grew", target_text="anything",
        gold_label=0, concepts=[{"id": "c", "hin": "energy", "ben": None}],
        lang_src="hin", lang_tgt="ben", max_len=32,
    )
    assert result.status == "not_applicable"


def test_not_applicable_when_target_span_cannot_be_recovered(monkeypatch):
    fake = _FakeCompute([
        ([5.0, 5.0], _ws_offsets("energy sector"), 0.01),
        ([5.0, 5.0], [], 0.01),  # empty offsets -> span never found
    ])
    monkeypatch.setattr(S, "compute_salience", fake)
    result = S.diagnose_instance(
        loaded=None, tokenizer=None, source_text="energy sector", target_text="urja sector",
        gold_label=0, concepts=[{"id": "c", "hin": "energy", "ben": "urja"}],
        lang_src="hin", lang_tgt="ben", max_len=32,
    )
    assert result.status == "not_applicable"


def test_not_fired_when_divergence_below_threshold(monkeypatch):
    source, target = "energy sector grew", "urja sector grew"
    fake = _FakeCompute([
        ([5.0, 2.5, 2.5], _ws_offsets(source), 0.01),   # share 0.5
        ([4.0, 3.0, 3.0], _ws_offsets(target), 0.01),   # share 0.4 -> divergence 0.1
    ])
    monkeypatch.setattr(S, "compute_salience", fake)
    result = S.diagnose_instance(
        loaded=None, tokenizer=None, source_text=source, target_text=target,
        gold_label=0, concepts=[{"id": "c", "hin": "energy", "ben": "urja"}],
        lang_src="hin", lang_tgt="ben", max_len=32,
    )
    assert result.status == "not_fired"
    assert result.divergence == pytest.approx(0.1)


def test_not_fired_when_target_more_salient_than_source(monkeypatch):
    source, target = "energy sector grew", "urja sector grew"
    fake = _FakeCompute([
        ([2.0, 4.0, 4.0], _ws_offsets(source), 0.01),   # share 0.2
        ([6.0, 2.0, 2.0], _ws_offsets(target), 0.01),   # share 0.6 -> divergence -0.4
    ])
    monkeypatch.setattr(S, "compute_salience", fake)
    result = S.diagnose_instance(
        loaded=None, tokenizer=None, source_text=source, target_text=target,
        gold_label=0, concepts=[{"id": "c", "hin": "energy", "ben": "urja"}],
        lang_src="hin", lang_tgt="ben", max_len=32,
    )
    assert result.status == "not_fired"
    assert result.divergence < 0


# --------------------------------------------------------------------------
# Slow: the real model + Captum, on 5 sampled pairs
# --------------------------------------------------------------------------


@pytest.mark.slow
def test_real_ig_renders_for_five_sampled_pairs():
    pytest.importorskip("captum")
    pytest.skip(
        "needs a frozen Phase 2/3 checkpoint (src.inference.load_frozen_model) and "
        "a populated configs/esg_terms.json -- run manually once both exist, per "
        "T-604's done bar (salience maps for 5 sampled pairs, convergence delta "
        "checked and within tolerance)"
    )
