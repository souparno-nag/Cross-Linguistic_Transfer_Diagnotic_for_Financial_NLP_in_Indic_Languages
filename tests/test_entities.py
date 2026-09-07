"""Tests for T-105. §10 requires >=5 positive and >=5 negative fixtures per check.

The positive cases matter more than usual here: this checker's failure mode is
crying wolf. A naive version reported 17% numeral loss on task 1 where the true
figure is 1.7%, which would have ranked task 1 worst in T-111.
"""

import pytest

from src.entities import compare, corruption_flags, extract

# --- scale folding: these pairs are the SAME quantity ----------------------

EQUIVALENT = [
    ("১০০ মিলিয়ন", "10 करोड़"),          # 100 million == 10 crore
    ("50 हजार", "50,000"),                # scale word vs written out
    ("50 हजार", "50, 000"),               # postprocessing adds a space
    ("২ কোটি", "20000000"),               # crore expanded
    ("৮ কোটি টাকা", "8 करोड़ रुपये"),      # crore across scripts
    ("45 लाख", "45 లక్షల"),                # lakh across scripts
    ("1 बिलियन", "100 करोड़"),             # billion == 100 crore
]


@pytest.mark.parametrize("left,right", EQUIVALENT)
def test_equivalent_quantities_are_preserved(left, right):
    assert compare(left, right).preserved, f"{left!r} vs {right!r}"


# --- genuine failures ------------------------------------------------------

LOSSES = [
    ("1,400 होटल हैं", "होटल हैं"),                     # number dropped
    ("38 प्रतिशत", "शतांश बढ़ा"),                        # number dropped
    ("₹2.5 लाख", "2.5 লাখ"),                            # currency dropped
    ("5 करोड़ रुपये", "5 करोड़"),                        # currency dropped
    ("38 प्रतिशत बढ़ा", "38 बढ़ा"),                      # percent marker dropped
    ("10 करोड़", "10 लाख"),                             # value altered
    ("2019 में 38 प्रतिशत", "2019 में"),                 # one of two dropped
]


@pytest.mark.parametrize("source,translation", LOSSES)
def test_real_losses_are_caught(source, translation):
    result = compare(source, translation)
    assert not result.preserved, f"{source!r} -> {translation!r}"
    assert "entity_loss" in result.flags()


# --- separators ------------------------------------------------------------


def test_thousands_separator_versus_list_separator():
    """A comma separates thousands only when exactly three digits follow."""
    assert extract("1,400").values == [1400.0]
    assert extract("50,000").values == [50000.0]
    # Two numbers, not one: 22 and 2019.
    assert sorted(extract("जुलाई 22,2019").values) == [22.0, 2019.0]
    assert compare("जुलाई 22, 2019", "জুলাই 22,2019").preserved


def test_indic_digits_fold_to_ascii():
    assert extract("১০").values == [10.0]
    assert extract("౧౨౩").values == [123.0]
    assert compare("১০ কোটি", "10 करोड़").preserved


# --- currency and percent across scripts -----------------------------------


def test_currency_matches_across_scripts():
    """The same unit written in four scripts is not a lost symbol."""
    for written in ("₹5 लाख", "5 লাখ টাকা", "రూ 5 లక్షల", "5 ലക്ഷം രൂപ"):
        assert "INR" in extract(written).currencies, written


def test_percent_matches_across_scripts():
    for written in ("38%", "38 प्रतिशत", "38 শতাংশ", "38 శాతం", "38 ശതമാനം"):
        assert extract(written).percentages >= 1, written
    assert compare("38 प्रतिशत", "38 শতাংশ").preserved


# --- corruption ------------------------------------------------------------

CORRUPT = [
    "पी. आई. बी हिन्दी (<আই. ডি. 1>)",   # unrestored placeholder
    "प्रति वर्ष <ID 2> टन",
    "रीयू _ यू09बीसी सेल",                # escape leak, Devanagari
    "റിയൽ/യു09ബിസി സെൽ",                 # escape leak, Malayalam
    "গড় ইউ09 ফলন",                       # escape leak, Bengali
    "విలువ యు09 శాతం",                    # escape leak, Telugu
]


@pytest.mark.parametrize("text", CORRUPT)
def test_corruption_is_detected_in_every_target_script(text):
    assert corruption_flags(text), text


CLEAN = [
    "কোম্পানিটি 10 কোটি টাকা বিনিয়োগ করেছে।",
    "బీఎస్‌ఈలో మిడ్‌ క్యాప్స్‌ 0.3 శాతం బలహీనపడింది.",
    "കാർബൺ ഫുട്പ്രിന്റ് കുറയ്ക്കുന്നു",
    "कंपनी ने ₹2.5 लाख का निवेश किया",
    "2019 में हवाई यात्रा 38 प्रतिशत थी।",
    "মাতিল ম্যানুকান (১৯১৪-২০০১)",
]


@pytest.mark.parametrize("text", CLEAN)
def test_clean_text_is_not_flagged_as_corrupt(text):
    assert corruption_flags(text) == [], text


def test_corrupted_rows_are_not_blamed_on_numerals():
    """Corruption and numeral loss are different failures (§3.4)."""
    result = compare("6-555 टन प्रति वर्ष", "প্রতি বছর <আই. ডি. 1> টন")
    assert "placeholder_leak" in result.flags()
    assert "entity_loss" not in result.flags(), (
        "content lost to the placeholder bug must not be attributed to numeral handling"
    )


def test_dropped_and_altered_are_reported_separately():
    """A missing figure and a wrong figure are different failures."""
    dropped = compare("10 करोड़ और 5 लाख", "10 करोड़")
    assert dropped.dropped and not dropped.added
    altered = compare("10 करोड़", "12 करोड़")
    assert altered.dropped and altered.added
