"""Morphological masking module (CLAUDE4.md T-604, this build's T-603) —
`morphological_masking`.

Fires when bound morphology masks a curated ESG root term: the term is
present in the target sentence only as a fuzzy-matched stem carrying an
attached suffix, never as an intact, unbound word. A three-rung fallback
ladder, per CLAUDE4.md's own spike:

1. Indic NLP Library's `UnsupervisedMorphAnalyzer` (Morfessor-based), which
   ships pretrained models for all four target languages (hi/bn/te/ml) —
   confirmed by directory listing in `anoopkunchukuttan/indic_nlp_resources`
   during Phase 6 planning; still to be confirmed by actually running it
   (this module degrades to rung 2 rather than trusting an unconfirmed API
   shape blindly).
2. A curated per-language case-marker / postposition suffix list,
   longest-suffix-first stripping. Used whenever rung 1 fails to load or run
   for a language — an infra failure, not "the term is absent" — so it is
   also what every fast test exercises when `indic-nlp-library` is not
   installed.
3. Declared `unavailable` for a language only if rung 2 has no suffix list
   for it either; those instances fall through to the next precedence level
   (`terminology_gap`), per CLAUDE4.md's "never drop an instance" hard rule.

Stanza is not attempted at all: Malayalam has no Universal Dependencies
treebank, and the only Malayalam Stanza package on Hugging Face is NER-only,
not a morphological analyser — confirmed during Phase 6 planning research.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

LANG_CODES = {"hin": "hi", "ben": "bn", "tel": "te", "mal": "ml"}

# Rung 2: curated case-marker / postposition suffixes. Sorted longest-first
# per language at import time (below), not by hand here, so a longer, more
# specific suffix is always tried before a shorter one that might also
# match a prefix of it -- codepoint length is a good enough proxy for this
# small, curated list, and sorting mechanically removes the risk of the
# list silently drifting out of order as entries are added.
_RAW_SUFFIXES: dict[str, tuple[str, ...]] = {
    "hin": ("में", "से", "को", "की", "का", "के", "पर", "ने"),
    "ben": ("দের", "গুলো", "টির", "এর", "কে", "তে", "র", "এ"),
    "tel": ("లోని", "నుండి", "తోని", "లో", "కి", "ను", "తో", "కు"),
    "mal": ("യിലെ", "ിന്റെ", "ോടെ", "ിൽ", "ൽ", "ിന്", "ോട്", "ുടെ"),
}
SUFFIXES: dict[str, tuple[str, ...]] = {
    lang: tuple(sorted(suffixes, key=len, reverse=True)) for lang, suffixes in _RAW_SUFFIXES.items()
}

PUNCT_RE = re.compile(r"[।॥,.!?;:\"'()\[\]—-]")


def _clean_word(word: str) -> str:
    return PUNCT_RE.sub("", word).strip()


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance — fuzzy-matches a lexicon term against a stem
    across the small spelling shifts sandhi and Morfessor segmentation
    boundaries introduce."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[-1]


def fuzzy_match(term: str, candidate: str, max_distance: int = 2) -> bool:
    """Plain edit-distance test. Kept for its tests; **not** the match rule
    (see `stem_matches`)."""
    return edit_distance(term, candidate) <= max_distance


# Shortest term (in codepoints) allowed any spelling change at all.
MIN_TERM_FOR_END_CHANGE = 4


def stem_matches(term: str, stem: str) -> bool:
    """Is `stem` this lexicon term, once its inflectional suffix is removed?

    The match rule, revised 2026-10-11. The first rule (`fuzzy_match`, edit
    distance <= 2 at any length) let a 2-codepoint term match *any* 2-codepoint
    stem: "जल" (water) matched कर (do), हो (be), हम (we), and 96% of all
    morphological-masking labels in the 2026-10-09 run were such fuzzy hits.
    That was a correctness bug — the check no longer tested term identity — not
    a threshold to tune (CLAUDE4.md "Scope decisions" / T-604).

    Ordinary inflection counts as masking (decided 2026-10-11), so the rule
    accepts exactly the stem changes inflection makes, and nothing else:

    * exact: `stem == term`;
    * for terms of at least `MIN_TERM_FOR_END_CHANGE` codepoints only:
      - the final codepoint **dropped** — Telugu's oblique stem
        (ఉద్గారాలు -> ఉద్గారాల+ను), or a segment boundary one codepoint early
        (সম্পদ -> সম্প+দের);
      - one **vowel sign added** at the end — the Bengali genitive splitting as
        নির্গমনে+র, Malayalam പരിസ്ഥിതിയ+ിൽ (combining marks, categories Mn/Mc).

    A substitution is never accepted: সম্পদ (resource) against সম্পর্ (the stem of
    সম্পর্কে, "about") was the one remaining false positive of a looser version.
    Short terms (জল, जल) must match exactly — at two codepoints any change makes
    another word.
    """
    if stem == term:
        return True
    if len(term) < MIN_TERM_FOR_END_CHANGE:
        return False
    if stem == term[:-1]:
        return True
    return (
        len(stem) == len(term) + 1
        and stem.startswith(term)
        and unicodedata.category(stem[-1]) in ("Mn", "Mc")
    )


def strip_suffix(word: str, lang: str) -> tuple[str, str]:
    """Rung 2: the longest matching suffix stripped from `word`, else the
    word is returned whole with an empty suffix (nothing to strip)."""
    for suffix in SUFFIXES.get(lang, ()):
        if word.endswith(suffix) and len(word) > len(suffix):
            return word[: -len(suffix)], suffix
    return word, ""


@dataclass
class MorphAnalyzer:
    """Wraps rung 1 (Indic NLP Library) with the rung-2 fallback baked in."""

    lang: str
    _library: object = field(default=None, init=False, repr=False)
    _cache: dict = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self):
        self._try_load_library()

    def _try_load_library(self) -> None:
        code = LANG_CODES.get(self.lang)
        if code is None:
            return
        try:
            from indicnlp import common  # noqa: PLC0415
            from indicnlp.morph.unsupervised_morph import UnsupervisedMorphAnalyzer  # noqa: PLC0415

            # The library does not read $INDIC_RESOURCES_PATH on import -- it
            # stays an empty string until `common.init()` is called
            # explicitly, and every model load resolves against it. Found
            # live during Phase 6's spike: without this call,
            # UnsupervisedMorphAnalyzer fails on every language even when
            # the environment variable and the model files are both
            # correctly in place, and the failure looks identical to a
            # genuinely missing resource -- silently permanent rung-2
            # fallback. `init()` is safe to call more than once per
            # process (it is a no-op once the module global is set).
            common.init()
            self._library = UnsupervisedMorphAnalyzer(code)
        except Exception:
            # Missing package, missing INDIC_RESOURCES_PATH, no model for
            # this language -- any of these is an infra failure, not "the
            # term isn't there", so this falls to rung 2 rather than raising.
            self._library = None

    @property
    def available(self) -> bool:
        return self._library is not None or self.lang in SUFFIXES

    @property
    def rung(self) -> str:
        if self._library is not None:
            return "library"
        if self.lang in SUFFIXES:
            return "suffix_rules"
        return "unavailable"

    def segment_word(self, word: str) -> tuple[str, str]:
        """`(stem, suffix)` for one cleaned word, via rung 1 else rung 2.

        Rung 1's exact return shape needs confirming against the real
        library (this module's live-spike done gate) — `morph_analyze` is
        documented to return the word's Morfessor segments space-joined,
        stem first. Any deviation from that shape degrades to rung 2 rather
        than raising, so an API surprise cannot silently corrupt a
        diagnosis.
        """
        # Segmentation depends on the word alone, and `check_instance` asks for
        # the same word once per lexicon concept: memoise, or a full re-run makes
        # ~19M Morfessor calls for ~40k rows.
        if word not in self._cache:
            self._cache[word] = self._segment_uncached(word)
        return self._cache[word]

    def _segment_uncached(self, word: str) -> tuple[str, str]:
        if self._library is not None:
            try:
                segmented = self._library.morph_analyze(word)
                parts = [p for p in segmented.split(" ") if p]
                if len(parts) >= 2:
                    return parts[0], "".join(parts[1:])
                if len(parts) == 1:
                    return parts[0], ""
            except Exception:
                pass
        return strip_suffix(word, self.lang)


@dataclass
class MorphResult:
    status: str  # "fired" | "not_fired" | "unavailable"
    evidence: dict | None = None


def check_instance(analyzer: MorphAnalyzer, text: str, concepts: list[dict]) -> MorphResult:
    """Does any lexicon concept's surface form for `analyzer.lang` appear in
    `text` only as a masked stem, never intact?

    A concept missing a surface form for this language (still `null` —
    `configs/esg_terms.json` before `scripts.t603_esg_terms` has run) is
    skipped for this instance, never treated as a non-match.
    """
    if not analyzer.available:
        return MorphResult(status="unavailable")

    words = [w for w in (_clean_word(w) for w in text.split()) if w]
    word_set = set(words)

    for concept in concepts:
        # Terms are cleaned like the words they are compared with: the
        # machine-derived lexicon carries two stray sentence marks (ben
        # "বর্জ্য।", tel "వాతావరణం."), which would otherwise never match a word.
        term = _clean_word(concept.get(analyzer.lang) or "")
        if not term:
            continue
        if term in word_set:
            continue  # intact -- visible, not masked
        for word in words:
            stem, suffix = analyzer.segment_word(word)
            if suffix and stem_matches(term, stem):
                return MorphResult(
                    status="fired",
                    evidence={
                        "term": term,
                        "word": word,
                        "stem": stem,
                        "suffix": suffix,
                        "concept": concept["id"],
                        "rung": analyzer.rung,
                    },
                )
    return MorphResult(status="not_fired")
