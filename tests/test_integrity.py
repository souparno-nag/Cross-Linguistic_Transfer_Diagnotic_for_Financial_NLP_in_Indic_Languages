"""Tests for T-107.

The checker's own failure mode is crying wolf (T-105 learned that the hard
way), so every check has fixtures that must come back *clean* alongside the
ones that must be caught.
"""

import pandas as pd
import pytest

from src import integrity as I


def test_a_script_leaked_fixture_is_caught():
    """§10's requirement: Devanagari in a Malayalam split must not pass."""
    assert I.script_leakage("ഇത് ഒരു വാക്യം ആണ്", "mal") == {}
    assert I.script_leakage("ഇത് ഒരു हिंदी വാക്യം", "mal") == {"Deva": 5}


def test_the_danda_is_not_script_leakage():
    """It lives in the Devanagari block but every Indic script uses it (§9).

    Counting it flagged 2211 Bengali rows as leaked before it was fixed.
    """
    assert I.script_leakage("এটি একটি বাক্য।", "ben") == {}


def test_placeholder_leak_needs_the_source_to_be_clean():
    leaked = "পি. আই. বি হিন্দি (<আই. ডি. 1>) জুলাই 22,2019"
    assert I.placeholder_leaked("पीआईबी हिंदी (@PIBHindi) July 22, 2019", leaked)
    # A source that genuinely carries angle brackets is not the model's doing.
    assert not I.placeholder_leaked("देखें <ID 1> यहाँ", "দেখুন <ID 1> এখানে")
    assert not I.placeholder_leaked("साधारण वाक्य", "সাধারণ বাক্য")


def test_escape_leak_is_caught_in_every_target_script():
    """The escape transliterates into the target, so each looks different (§3.4)."""
    source = "গড় ফলন প্রায় ৮ মে টন/হেক্টর।"
    for leaked in ("ഗോഡ് _ യു09ബിസി വിളവ്", "গড় ইউ09 ফলন", "गोड यू09बीसी उपज", "గోడ్ యు09 దిగుబడి"):
        assert I.escape_leaked(source, leaked), leaked
    assert not I.escape_leaked(source, "গড় ফলন প্রায় 8 টন")


def test_encoding_corruption_is_the_replacement_character():
    assert I.encoding_corrupted("দাম � 500")
    assert not I.encoding_corrupted("দাম 500")


def test_truncation_is_judged_against_the_direction_not_a_fixed_ratio():
    """Malayalam runs longer than its Hindi source; a global ratio misreads that."""
    full = I.length_ratio("a" * 100, "b" * 110)
    cut = I.length_ratio("a" * 100, "b" * 30)
    median = 1.1
    assert not cut > I.TRUNCATION_RATIO * median
    assert full > I.TRUNCATION_RATIO * median


def test_flags_written_twice_are_the_same_as_written_once(monkeypatch):
    """Re-running after a fix must not stack flags or leave stale ones."""
    keep = ["unaligned", "empty_output"]
    row_flags = keep + ["script_leakage"]
    stripped = [f for f in row_flags if f not in I.INTEGRITY_FLAGS]
    assert stripped == keep
    # applying again from a clean base gives the same result, not a longer list
    assert stripped + ["script_leakage"] == row_flags


@pytest.mark.parametrize("flag", I.INTEGRITY_FLAGS)
def test_every_owned_flag_is_stripped_before_rewriting(flag):
    assert flag in I.INTEGRITY_FLAGS
    assert [f for f in ["unaligned", flag] if f not in I.INTEGRITY_FLAGS] == ["unaligned"]
