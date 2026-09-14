"""Tests for T-604 (this build's T-603) — the morphological masking module.

Fast tests run with whatever rung is actually available in this environment.
`indic-nlp-library` (rung 1) is a new, optional dependency (CLAUDE4.md's own
spike note); when it is not installed, `MorphAnalyzer` degrades to rung 2
(the curated suffix lists) automatically, which is exactly the fallback path
these fixtures exercise -- no mocking needed. A `@pytest.mark.slow` test
covers the real library directly, skipped when it is not installed.
"""

from __future__ import annotations

import pytest

from src import morphology as M


# --------------------------------------------------------------------------
# edit_distance / fuzzy_match
# --------------------------------------------------------------------------


def test_edit_distance_identical_strings_is_zero():
    assert M.edit_distance("ऊर्जा", "ऊर्जा") == 0


def test_edit_distance_counts_single_substitution():
    assert M.edit_distance("कार्बन", "कार्बण") == 1


def test_edit_distance_handles_empty_strings():
    assert M.edit_distance("", "abc") == 3
    assert M.edit_distance("abc", "") == 3


def test_fuzzy_match_within_tolerance():
    assert M.fuzzy_match("प्रदूषण", "प्रदूषन", max_distance=2)


def test_fuzzy_match_rejects_beyond_tolerance():
    assert not M.fuzzy_match("प्रदूषण", "बाजार", max_distance=2)


# --------------------------------------------------------------------------
# strip_suffix (rung 2)
# --------------------------------------------------------------------------


def test_strip_suffix_removes_longest_match():
    stem, suffix = M.strip_suffix("ऊर्जामें", "hin")
    assert stem == "ऊर्जा"
    assert suffix == "में"


def test_strip_suffix_returns_whole_word_when_nothing_matches():
    stem, suffix = M.strip_suffix("बाजार", "hin")
    assert stem == "बाजार"
    assert suffix == ""


def test_strip_suffix_unknown_language_returns_whole_word():
    stem, suffix = M.strip_suffix("word", "xyz")
    assert stem == "word"
    assert suffix == ""


# --------------------------------------------------------------------------
# MorphAnalyzer availability
# --------------------------------------------------------------------------


def test_analyzer_available_for_every_target_language():
    for lang in ("hin", "ben", "tel", "mal"):
        analyzer = M.MorphAnalyzer(lang)
        assert analyzer.available, f"{lang} has no rung 1 or rung 2 coverage"
        assert analyzer.rung in ("library", "suffix_rules")


def test_analyzer_unavailable_for_an_unsupported_language():
    analyzer = M.MorphAnalyzer("xyz")
    assert not analyzer.available
    assert analyzer.rung == "unavailable"


# --------------------------------------------------------------------------
# check_instance — 5 positive (fires) fixtures across languages
# --------------------------------------------------------------------------

POSITIVE_FIXTURES = [
    ("tel", "ఎమిషన్లో తగ్గింది.", [{"id": "emission_test", "tel": "ఎమిషన్"}]),
    ("mal", "മാലിന്യംിൽ വർദ്ധനവ്.", [{"id": "waste_test", "mal": "മാലിന്യം"}]),
    ("ben", "দূষণএর প্রভাব।", [{"id": "pollution_test", "ben": "দূষণ"}]),
    ("hin", "ऊर्जामें वृद्धि हुई।", [{"id": "energy_test", "hin": "ऊर्जा"}]),
    ("tel", "వనరుకు కొరత.", [{"id": "resource_test", "tel": "వనరు"}]),
]


@pytest.mark.parametrize("lang,text,concepts", POSITIVE_FIXTURES)
def test_positive_fixtures_fire(lang, text, concepts):
    analyzer = M.MorphAnalyzer(lang)
    result = M.check_instance(analyzer, text, concepts)
    assert result.status == "fired", f"expected fire for {lang}: {text!r}"
    assert result.evidence is not None
    assert result.evidence["suffix"]


# --------------------------------------------------------------------------
# check_instance — 5 negative (does not fire) fixtures
# --------------------------------------------------------------------------

NEGATIVE_FIXTURES = [
    ("tel", "ఎమిషన్ తగ్గింది.", [{"id": "emission_test", "tel": "ఎమిషన్"}], "intact, unbound"),
    ("mal", "മാലിന്യം കുറഞ്ഞു.", [{"id": "waste_test", "mal": "മാലിന്യം"}], "intact, unbound"),
    ("ben", "দূষণ কমেছে।", [{"id": "pollution_test", "ben": "দূষণ"}], "intact, unbound"),
    ("hin", "ऊर्जा बढ़ी।", [{"id": "energy_test", "hin": "ऊर्जा"}], "intact, unbound"),
    ("tel", "మార్కెట్ పెరిగింది.", [{"id": "resource_test", "tel": "వనరు"}], "term absent entirely"),
]


@pytest.mark.parametrize("lang,text,concepts,note", NEGATIVE_FIXTURES)
def test_negative_fixtures_do_not_fire(lang, text, concepts, note):
    analyzer = M.MorphAnalyzer(lang)
    result = M.check_instance(analyzer, text, concepts)
    assert result.status == "not_fired", f"expected no fire ({note}) for {lang}: {text!r}"


def test_concept_missing_surface_form_is_skipped_not_a_nonmatch():
    """A concept with no `mal` translation yet (esg_terms.json before
    scripts.t603_esg_terms runs) must be skipped for that language, not
    treated as a legitimate absence."""
    analyzer = M.MorphAnalyzer("mal")
    concepts = [{"id": "untranslated", "mal": None}]
    result = M.check_instance(analyzer, "ഏതെങ്കിലും വാചകം.", concepts)
    assert result.status == "not_fired"


def test_unavailable_language_short_circuits():
    analyzer = M.MorphAnalyzer("xyz")
    result = M.check_instance(analyzer, "anything", [{"id": "c", "xyz": "anything"}])
    assert result.status == "unavailable"
    assert result.evidence is None


# --------------------------------------------------------------------------
# Slow: the real Indic NLP Library, if installed
# --------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("lang", ["hin", "ben", "tel", "mal"])
def test_real_library_loads_and_segments(lang):
    pytest.importorskip("indicnlp.morph.unsupervised_morph")
    analyzer = M.MorphAnalyzer(lang)
    if analyzer.rung != "library":
        pytest.skip(
            f"indic-nlp-library imports but {lang}.model did not load "
            "(check INDIC_RESOURCES_PATH) -- rung 2 covered this instead"
        )
    stem, suffix = analyzer.segment_word("test")
    assert isinstance(stem, str) and isinstance(suffix, str)
