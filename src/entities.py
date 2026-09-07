"""T-105 — do financial entities survive translation?

Numerals are the semantic core of this domain, so this is the check the project
exists to make. It is also the check most easily got wrong, in three ways that
T-104 measured on real output:

* **Scale words.** `১০০ মিলিয়ন` (100 million) and `10 करोड़` (10 crore) are the
  same quantity written two ways. Comparing digits calls that a loss.
* **Thousands separators.** A comma separates thousands only when exactly three
  digits follow. `जुलाई 22, 2019` becoming `জুলাই 22,2019` is two numbers, not
  the single number 22,2019.
* **Corrupted rows.** Output mangled by the placeholder or escape bugs of §3.4
  lost content for reasons unrelated to numeral handling. Counting it as
  numeral loss blames the wrong cause.

A naive digit comparison reported 17% loss on task 1 where the true figure is
1.7% — enough to rank task 1 worst in T-111 when it is among the better ones.

Everything here works on text alone and holds no opinion about the corpus
schema, so T-107 can reuse :func:`corruption_flags` without importing anything
heavier.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .unicode_ranges import to_ascii_digits

# Multipliers, written as they appear in each language. Order matters when one
# term is a prefix of another, so longer forms are matched first.
SCALE_WORDS: dict[str, float] = {
    # Indic 'thousand'
    "हजार": 1e3, "हज़ार": 1e3, "হাজার": 1e3, "వేల": 1e3, "వేయి": 1e3,
    "ఆయిరం": 1e3, "ആയിരം": 1e3, "thousand": 1e3,
    # lakh = 10^5
    "लाख": 1e5, "লাখ": 1e5, "লক্ষ": 1e5, "లక్షల": 1e5, "లక్ష": 1e5,
    "ലക്ഷം": 1e5, "lakh": 1e5, "lakhs": 1e5,
    # crore = 10^7
    "करोड़": 1e7, "करोड": 1e7, "কোটি": 1e7, "కోట్ల": 1e7, "కోటి": 1e7,
    "കോടി": 1e7, "crore": 1e7, "crores": 1e7,
    # Western scales
    "मिलियन": 1e6, "মিলিয়ন": 1e6, "మిలియన్": 1e6, "ദശലക്ഷം": 1e6,
    "million": 1e6,
    "बिलियन": 1e9, "বিলিয়ন": 1e9, "బిలియన్": 1e9, "ബില്യൺ": 1e9,
    "अरब": 1e9, "billion": 1e9,
    "ट्रिलियन": 1e12, "ট্রিলিয়ন": 1e12, "trillion": 1e12,
}

# Currency, normalised to a single token per unit so `₹`, `रु` and `రూ` compare
# equal. A translation that switches script must not read as a lost symbol.
CURRENCY: dict[str, str] = {
    "₹": "INR", "rs": "INR", "rs.": "INR", "inr": "INR",
    "रुपये": "INR", "रुपए": "INR", "रु": "INR", "रुपया": "INR",
    "টাকা": "INR", "রুপি": "INR", "রুপী": "INR",
    "రూపాయల": "INR", "రూపాయి": "INR", "రూ": "INR",
    "രൂപ": "INR",
    "$": "USD", "usd": "USD", "डॉलर": "USD", "ডলার": "USD",
    "డాలర్": "USD", "ഡോളർ": "USD", "dollar": "USD", "dollars": "USD",
    "€": "EUR", "euro": "EUR", "यूरो": "EUR", "ইউরো": "EUR",
    "£": "GBP", "pound": "GBP", "पाउंड": "GBP", "পাউন্ড": "GBP",
}

PERCENT_WORDS = (
    "%", "प्रतिशत", "फीसदी", "শতাংশ", "শতকরা", "శాతం", "ശതമാനം", "percent",
)

# A comma is a thousands separator only when exactly three digits follow it.
# The space is not optional decoration: IndicProcessor's postprocessing emits
# `50, 000`, which without it parses as 50 and 0 rather than 50000.
THOUSANDS_SEPARATOR = re.compile(r"(?<=\d),\s*(?=\d{3}(?!\d))")
NUMBER = re.compile(r"\d+(?:\.\d+)?")

# §3.4's two corruption modes. Both leave fluent-looking text.
PLACEHOLDER_LEAK = re.compile(r"<[^>]{1,40}>")
# `u09bc` transliterates into the *target* script, so each one looks different:
# `യു09ബിസി` (Malayalam), `ইউ09` (Bengali), `यू09बीसी` (Devanagari),
# `యు09` (Telugu). Covering only two of them undercounts the corruption — which
# it did, until a Hindi-target row was found parsing `यू09बीसी` as the number 9.
ESCAPE_LEAK = re.compile(
    r"u09[0-9a-f]{2}|u093[0-9a-f]|യു[0-9]|ইউ[0-9]|यू[0-9]|యు[0-9]",
    re.IGNORECASE,
)

# How far after a number to look for its scale word. Wide enough for an
# intervening currency word, narrow enough not to grab the next clause.
SCALE_WINDOW = 12


def corruption_flags(text: str) -> list[str]:
    """Detect the §3.4 output corruptions. Shared with T-107."""
    flags = []
    if PLACEHOLDER_LEAK.search(str(text)):
        flags.append("placeholder_leak")
    if ESCAPE_LEAK.search(str(text)):
        flags.append("escape_leak")
    return flags


@dataclass
class Entities:
    """Financial entities found in one string."""

    values: list[float] = field(default_factory=list)
    currencies: list[str] = field(default_factory=list)
    percentages: int = 0

    def as_multiset(self) -> list[float]:
        return sorted(self.values)


def _scale_after(text: str, position: int) -> tuple[float, int]:
    """The scale word just after `position`, if any: (multiplier, length)."""
    window = text[position : position + SCALE_WINDOW]
    best = (1.0, 0)
    for word, multiplier in SCALE_WORDS.items():
        index = window.find(word)
        # Must follow the number closely, separated only by spaces or currency.
        if index != -1 and not any(ch.isdigit() for ch in window[:index]):
            if len(word) > best[1]:
                best = (multiplier, len(word))
    return best


def extract(text: str) -> Entities:
    """Numbers with their scale applied, currencies, and percentage markers.

    `50 हजार` yields 50000, not 50 — the scale word is folded into the value so
    it compares equal to a translation that writes `50,000` in full.
    """
    folded = to_ascii_digits(text)
    folded = THOUSANDS_SEPARATOR.sub("", folded)
    lowered = folded.lower()

    values = []
    for match in NUMBER.finditer(folded):
        number = float(match.group())
        multiplier, _ = _scale_after(folded, match.end())
        values.append(number * multiplier)

    currencies = []
    for token, code in CURRENCY.items():
        if token in lowered:
            currencies.append(code)

    percentages = sum(lowered.count(word) for word in PERCENT_WORDS)
    return Entities(
        values=values, currencies=sorted(set(currencies)), percentages=percentages
    )


@dataclass
class Comparison:
    """What survived translation, and what did not."""

    preserved: bool
    dropped: list[float] = field(default_factory=list)
    added: list[float] = field(default_factory=list)
    currency_lost: list[str] = field(default_factory=list)
    percent_lost: int = 0
    corrupted: list[str] = field(default_factory=list)

    def flags(self) -> list[str]:
        """Corpus flags for this row (§6)."""
        flags = list(self.corrupted)
        if not self.preserved and not self.corrupted:
            flags.append("entity_loss")
        return flags


def compare(source: str, translation: str, tolerance: float = 1e-6) -> Comparison:
    """Did the entities in `source` survive into `translation`?

    Values are compared as a multiset, so a repeated number must appear the
    same number of times. A *dropped* value and an *altered* one are reported
    separately: a missing figure and a wrong figure are different failures, and
    only the second is silently plausible to a reader.

    A corrupted row is reported as corrupted rather than as entity loss. Its
    content went missing for an unrelated reason, and attributing it to numeral
    handling would misdirect T-111's ranking.
    """
    corrupted = corruption_flags(translation)
    left, right = extract(source), extract(translation)

    remaining = list(right.values)
    dropped = []
    for value in left.values:
        match = next(
            (v for v in remaining if abs(v - value) <= tolerance * max(1.0, abs(value))),
            None,
        )
        if match is None:
            dropped.append(value)
        else:
            remaining.remove(match)

    currency_lost = [c for c in left.currencies if c not in right.currencies]
    percent_lost = max(0, left.percentages - right.percentages)

    preserved = not dropped and not remaining and not currency_lost and not percent_lost
    return Comparison(
        preserved=preserved,
        dropped=dropped,
        added=remaining,
        currency_lost=currency_lost,
        percent_lost=percent_lost,
        corrupted=corrupted,
    )
