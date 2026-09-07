"""Script and digit codepoint tables for the four corpus languages.

These are the tables in CLAUDE.md §9, in one place. The audit (T-102), the
entity checker (T-105) and the integrity checker (T-107) all need them, and
three hand-copied sets of codepoint ranges is how they drift apart.

Nothing here knows about pandas or the corpus schema — it is deliberately
dependency-free so every layer can import it.
"""

from __future__ import annotations

import unicodedata

# Language code -> script tag, in the FLORES-style pairing IndicTrans2 uses.
LANG_SCRIPT = {
    "hin": "Deva",
    "ben": "Beng",
    "tel": "Telu",
    "mal": "Mlym",
}

# Upstream IndicFinNLP spells languages out; we key on the three-letter code.
LANG_ALIASES = {
    "hindi": "hin",
    "bengali": "ben",
    "telugu": "tel",
    "malayalam": "mal",
}

# Script-neutral punctuation that lives inside the Devanagari block despite
# being shared across Indic scripts. Bengali, Telugu and Malayalam text all use
# the danda as a sentence terminator, so counting it as Devanagari would flag
# essentially every Bengali row as script leakage. Excluded from all script
# tallies.
SHARED_PUNCTUATION = frozenset({
    0x0964,  # DEVANAGARI DANDA
    0x0965,  # DEVANAGARI DOUBLE DANDA
})

# Script blocks, used by the leakage check. Inclusive bounds.
SCRIPT_BLOCKS = {
    "Deva": (0x0900, 0x097F),
    "Beng": (0x0980, 0x09FF),
    "Telu": (0x0C00, 0x0C7F),
    "Mlym": (0x0D00, 0x0D7F),
}

# Digit ranges. Each script's zero, so digit N is zero + N.
DIGIT_ZERO = {
    "Deva": 0x0966,
    "Beng": 0x09E6,
    "Telu": 0x0C66,
    "Mlym": 0x0D66,
    "ASCII": 0x0030,
}

DIGIT_BLOCKS = {script: (zero, zero + 9) for script, zero in DIGIT_ZERO.items()}

# Invisible joiners. They carry conjunct information in Indic scripts, so a
# count of zero is worth knowing: it means the source was written without them
# and any that appear post-translation came from the model.
ZWNJ = "‌"
ZWJ = "‍"

REPLACEMENT_CHAR = "�"


def normalise_lang(name: str) -> str:
    """Map 'bengali' or 'ben' onto the canonical three-letter code."""
    key = name.strip().lower()
    if key in LANG_SCRIPT:
        return key
    if key in LANG_ALIASES:
        return LANG_ALIASES[key]
    raise ValueError(f"unknown language {name!r}; extend LANG_ALIASES")


def script_of(lang: str) -> str:
    """Script tag expected for a language, e.g. 'ben' -> 'Beng'."""
    return LANG_SCRIPT[normalise_lang(lang)]


def script_counts(text: str) -> dict[str, int]:
    """Characters per script block. Scripts absent from the text are omitted."""
    counts: dict[str, int] = {}
    for char in text:
        code = ord(char)
        if code in SHARED_PUNCTUATION:
            continue
        for script, (low, high) in SCRIPT_BLOCKS.items():
            if low <= code <= high:
                counts[script] = counts.get(script, 0) + 1
                break
    return counts


def digit_counts(text: str) -> dict[str, int]:
    """Digits per numbering system, including ASCII."""
    counts: dict[str, int] = {}
    for char in text:
        code = ord(char)
        for script, (low, high) in DIGIT_BLOCKS.items():
            if low <= code <= high:
                counts[script] = counts.get(script, 0) + 1
                break
    return counts


# Folding every Indic digit onto ASCII is what lets a Bengali ১০ and a Telugu
# ౧౦ compare equal to 10. Built once from DIGIT_ZERO so the two never drift.
ASCII_DIGIT_TABLE = {
    zero + offset: str(offset)
    for script, zero in DIGIT_ZERO.items()
    if script != "ASCII"
    for offset in range(10)
}


def to_ascii_digits(text: str) -> str:
    """Rewrite Devanagari, Bengali, Telugu and Malayalam digits as ASCII."""
    return str(text).translate(ASCII_DIGIT_TABLE)


def foreign_scripts(text: str, lang: str) -> dict[str, int]:
    """Script-block characters that do not belong to `lang`.

    Latin, punctuation and digits are not scripts here — only the four Indic
    blocks are checked, so an English brand name in a Hindi sentence is not
    flagged but a Devanagari word in a Bengali sentence is.
    """
    expected = script_of(lang)
    return {s: n for s, n in script_counts(text).items() if s != expected}


def is_nfc(text: str) -> bool:
    """True when the text is already in NFC.

    Indic text that mixes normalisation forms compares unequal byte-wise while
    looking identical, which would silently break the join in §6 and every
    duplicate check.
    """
    return unicodedata.is_normalized("NFC", text)
