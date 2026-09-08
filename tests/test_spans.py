"""Tests for T-106's span recovery (§6.1).

The failure this guards against is a recovered span that points at the wrong
characters: `span_recovered` is True, the offsets look like data, and nothing
downstream can tell. So every positive case asserts the slice, not the flag.
"""

import pytest

from src.spans import parse_number, recover_span


@pytest.mark.parametrize(
    "raw, value",
    [
        ("24", 24.0),
        ("1,400", 1400.0),
        ("7,30,688", 730688.0),   # Indian grouping
        ("22,", 22.0),            # upstream trailing punctuation
        ("66.", 66.0),
        ("০.৩", 0.3),             # Bengali digits with a decimal point
    ],
)
def test_upstream_number_forms_all_parse(raw, value):
    assert parse_number(raw) == value


def test_unparseable_number_is_none():
    assert parse_number("") is None
    assert parse_number("साल") is None


@pytest.mark.parametrize(
    "translation, number",
    [
        ("এনপিসিআই দ্বারা ২৪ ঘন্টা সংদায় ব্যবস্থা", "24"),   # target-script digits
        ("NPCI-യുടെ 24 മണിക്കൂർ സംവിധാനം", "24"),           # ASCII digits
        ("మొత్తం 25,000 రూపాయలు", "25000"),                  # grouping added
        ("মোট 50, 000 টাকা", "50000"),                        # postprocessor's space
        ("মোট 7,30,688 টাকা", "730688"),                      # Indian grouping
    ],
)
def test_recovered_offsets_point_at_the_number(translation, number):
    found = recover_span(translation, number)
    assert found.recovered
    assert translation[found.start : found.end] == found.matched
    assert parse_number(found.matched) == parse_number(number)


def test_a_year_is_not_swallowed_by_the_preceding_number():
    """`22, 2019` is two numbers; parsing it as one would misplace the span."""
    text = "জুলাই 22, 2019 তারিখে"
    for number, expected in (("22", "22"), ("2019", "2019")):
        found = recover_span(text, number)
        assert found.recovered and found.matched == expected
        assert text[found.start : found.end] == expected


def test_missing_number_is_reported_not_guessed():
    found = recover_span("এই বাক্যে কোনো সংখ্যা নেই", "24")
    assert not found.recovered
    assert (found.start, found.end, found.matched) == (-1, -1, "")
    assert found.flags == ["span_not_recovered"]


def test_empty_translation_recovers_nothing():
    assert not recover_span("", "24").recovered


@pytest.mark.parametrize(
    "source, translation, number",
    [
        ("১০০ মিলিয়ন টাকার প্রকল্প", "10 करोड़ रुपये की परियोजना", "100"),
        ("50 हजार रुपये", "50,000 রুপি", "50"),
    ],
)
def test_a_rescaled_number_is_distinguished_from_a_lost_one(source, translation, number):
    """The quantity survived; only the digit token moved. Not the same failure."""
    found = recover_span(translation, number, source)
    assert not found.recovered
    assert "span_scale_shift" in found.flags


def test_repeated_value_is_flagged_rather_than_silently_chosen():
    found = recover_span("24 থেকে 24 পর্যন্ত", "24")
    assert found.recovered and "span_ambiguous" in found.flags
    assert found.start == 0


def test_unparseable_source_number_says_so():
    assert recover_span("কিছু 10 আছে", "n/a").flags == ["span_unparseable_source"]


def test_a_list_of_numbers_survives_losing_its_spaces():
    """`145, 146, 150` → `145,146,150` is three rule numbers, not one number.

    Translation drops the space the source used to disambiguate, and reading
    the commas as grouping made every member of such a list unrecoverable —
    the single biggest systematic gap in task 1's recovery.
    """
    text = "জি. এফ. আর নিয়ম 145,146,150 এবং 151 প্রযোজ্য হবে।"
    for number in ("145,", "146,", "150", "151"):
        found = recover_span(text, number, "जीएफआर नियम 145, 146, 150 और 151 लागू होंगे।")
        assert found.recovered, f"{number} not found"
        assert parse_number(found.matched) == parse_number(number)
        assert text[found.start : found.end] == found.matched


def test_the_list_fallback_does_not_break_a_thousands_separator():
    """`50 हजार` → `50, 000` must not put the span on the leading `50`.

    The digits there belong to the rescaled 50,000. A span on `50` would point
    at the wrong characters while claiming success, which is worse than
    reporting the number as not found.
    """
    found = recover_span("50, 000 টাকা অগ্রিম দেওয়া হয়েছিল।", "50", "50 हजार रुपये एडवांस दिए थे।")
    assert not found.recovered
    assert found.flags == ["span_not_recovered", "span_scale_shift"]


def test_grouped_number_still_wins_over_its_own_parts():
    """Seeking 25000 in `25,000` must match the whole, not stop at `25`."""
    found = recover_span("মোট 25,000 টাকা", "25,000")
    assert found.recovered and found.matched == "25,000"
