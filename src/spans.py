"""Re-locating a task-1 numeral inside its translation (T-106, §6.1).

Task 1 marks one number inside a sentence, addressed by character offsets into
the source string. Those offsets do not survive translation: the target
sentence is a different length in a different script, so carrying `start_posn`
across would point at the wrong characters while still looking like data. The
number has to be *found* again in the output.

Recovery can fail — the model may drop the number, rewrite it with a scale word
(`50 हजार` → `50,000`), or mangle it. That failure rate is a headline result
rather than an error (§6.1), so this module reports *why* a span was not
recovered instead of only that it was not.

Text only: nothing here knows about pandas or the corpus schema, so it can be
tested without a GPU or a corpus.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .entities import extract
from .unicode_ranges import to_ascii_digits

# A number, optionally written with grouping separators. Both groupings occur:
# Western `25,000` and Indian `7,30,688`, and IndicProcessor's postprocessing
# emits a space after the comma (`50, 000`). A group is only a group when it
# ends there — without the lookahead, `22, 2019` would match as one number
# spanning both, which is exactly the mis-parse T-105 was built to avoid.
NUMBER_WITH_GROUPS = re.compile(r"\d+(?:,\s?\d{2,3}(?!\d))*(?:\.\d+)?")

# Only these are removed when reading a matched string as a value. Folding
# Indic digits to ASCII is a character-for-character translation, so offsets
# into the folded string are also offsets into the original — which is what
# makes it safe to search the folded text and return the raw text's positions.
GROUPING = re.compile(r"[,\s]")

# Scale words fold into the value (T-105), so a number carrying one shows up in
# `extract` already multiplied. These are the multipliers that can separate the
# annotated digit token from the quantity it denotes.
SCALE_MULTIPLES = (1e3, 1e5, 1e6, 1e7, 1e9, 1e12)


def parse_number(text: str) -> float | None:
    """Read upstream's `number_english` as a value.

    Upstream is not uniform: `1,400`, `7,30,688` and the trailing-punctuation
    forms `22,` and `66.` all appear. There is no ambiguity inside a single
    number field, so every separator is simply dropped.
    """
    folded = GROUPING.sub("", to_ascii_digits(text)).strip()
    folded = folded.rstrip(".,")
    if not folded:
        return None
    try:
        return float(folded)
    except ValueError:
        return None


@dataclass
class Recovery:
    """Where the number ended up, or why it did not."""

    start: int = -1
    end: int = -1
    matched: str = ""
    recovered: bool = False
    flags: list[str] = field(default_factory=list)


def _candidates(translation: str) -> list[tuple[int, int, str, float]]:
    folded = to_ascii_digits(translation)
    found = []
    for match in NUMBER_WITH_GROUPS.finditer(folded):
        value = parse_number(match.group())
        if value is not None:
            found.append((match.start(), match.end(), translation[match.start():match.end()], value))
    return found


def _scale_shifted(source: str, translation: str, value: float) -> bool:
    """Did the number survive as the same quantity written another way?

    `১০০ মিলিয়ন` (100 million) becoming `10 करोड़` (10 crore) leaves no digit
    token to point at, but nothing was lost. Distinguishing that from a number
    the model simply dropped is the difference between a translation artefact
    and a translation failure, and T-111 ranks directions on it.

    The annotated number is read together with whatever scale word follows it
    in the source, giving the quantity it denotes; the row is a scale shift
    when that quantity is present in the translation.
    """
    def close(a: float, b: float) -> bool:
        return abs(a - b) <= 1e-6 * max(1.0, abs(a))

    quantities = [
        v
        for v in extract(source).values
        if close(v, value) or (value and v / value in SCALE_MULTIPLES)
    ]
    produced = extract(translation).values
    return any(close(q, t) for q in quantities for t in produced)


def recover_span(translation: str, number_english: str, source_text: str = "") -> Recovery:
    """Find `number_english` in `translation` and return its offsets.

    The match is by *value*, not by digit string: the target may write the
    number in its own script (`২৪`), in ASCII, or with grouping separators the
    source did not use, and all three are the same number. Offsets returned
    index `translation` itself, so `translation[start:end]` is the number as
    the target actually wrote it.

    When several occurrences carry the same value the first is taken and the
    row is flagged `span_ambiguous` — either occurrence is defensible, and
    silently picking one without saying so is not.
    """
    value = parse_number(number_english)
    if value is None:
        return Recovery(flags=["span_unparseable_source"])
    if not str(translation).strip():
        return Recovery(flags=["span_not_recovered"])

    matches = [
        c for c in _candidates(translation)
        if abs(c[3] - value) <= 1e-6 * max(1.0, abs(value))
    ]
    if not matches:
        flags = ["span_not_recovered"]
        if source_text and _scale_shifted(source_text, translation, value):
            flags.append("span_scale_shift")
        return Recovery(flags=flags)

    start, end, matched, _ = matches[0]
    flags = ["span_ambiguous"] if len(matches) > 1 else []
    return Recovery(start=start, end=end, matched=matched, recovered=True, flags=flags)
