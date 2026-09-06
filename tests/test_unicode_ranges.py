"""Tests for the §9 script and digit tables."""

import pytest

from src.unicode_ranges import (
    digit_counts,
    foreign_scripts,
    is_nfc,
    normalise_lang,
    script_counts,
    script_of,
)

HINDI = "2019 में, हवाई यात्रा हमारे कार्बन फुटप्रिंट का लगभग 38 प्रतिशत थी।"
BENGALI = "যা টাকায় ১০ কোটি টাকারও বেশি।"
TELUGU = "రుణాలను సమలేఖనం చేయడానికి ABN AMRO."
MALAYALAM = "കാർബൺ ഫുട്പ്രിന്റ് ൧൦ ശതമാനം."


def test_language_aliases_and_scripts():
    assert normalise_lang("Bengali") == "ben"
    assert normalise_lang("ben") == "ben"
    assert script_of("telugu") == "Telu"
    with pytest.raises(ValueError):
        normalise_lang("marathi")


@pytest.mark.parametrize(
    "text,lang,script",
    [(HINDI, "hin", "Deva"), (BENGALI, "ben", "Beng"), (TELUGU, "tel", "Telu"),
     (MALAYALAM, "mal", "Mlym")],
)
def test_native_text_has_no_foreign_script(text, lang, script):
    assert script_counts(text)[script] > 0
    assert foreign_scripts(text, lang) == {}


def test_danda_is_not_devanagari_leakage():
    """U+0964 is shared Indic punctuation, not a Devanagari intrusion.

    Counting it as Devanagari flagged 2211 of 2228 Bengali rows as leaked.
    """
    assert foreign_scripts("যা টাকায় বেশি।", "ben") == {}
    assert foreign_scripts("తెలుగు వాక్యం।", "tel") == {}


def test_real_leakage_is_caught():
    assert foreign_scripts("আমার नाम", "ben") == {"Deva": 3}
    # "আমার" is four codepoints: আ ম া র
    assert foreign_scripts("తెలుగు আমার", "tel") == {"Beng": 4}


def test_latin_and_punctuation_are_not_leakage():
    """An English brand name in Indic text is normal, not a script error."""
    assert foreign_scripts(TELUGU, "tel") == {}


def test_digit_systems_are_distinguished():
    assert digit_counts("১০ 10 १० ౧౦ ൧൦") == {
        "Beng": 2, "ASCII": 2, "Deva": 2, "Telu": 2, "Mlym": 2
    }


def test_nfc_detection():
    assert is_nfc("क्ष")
    assert not is_nfc("é".encode().decode() + "́" if False else "é")
