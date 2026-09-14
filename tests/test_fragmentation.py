"""Tests for T-602 (CLAUDE4.md's fragmentation ratio module).

Fast tests exercise the ratio/boundary arithmetic against a trivial stub
tokenizer (whitespace split) so they need no download. The `@pytest.mark.slow`
tests hand-verify 5 known-fragmenting and 5 non-fragmenting Hindi->target
sentence pairs against two real tokenizers (`indicbert-v2`, `xlm-r-base`),
matching T-602's own done criterion: "correct for at least two different
tokenizers."
"""

from __future__ import annotations

import pytest

from src import fragmentation as F
from src.data import get_tokenizer


class _WhitespaceTokenizer:
    """Deterministic stand-in: one token per whitespace-separated word."""

    def tokenize(self, text: str) -> list[str]:
        return text.split()


@pytest.fixture()
def stub():
    return _WhitespaceTokenizer()


# --------------------------------------------------------------------------
# Fast: ratio arithmetic and the fires() boundary
# --------------------------------------------------------------------------


def test_token_count_counts_whitespace_words(stub):
    assert F.token_count(stub, "one two three") == 3


def test_r_frag_is_target_over_source(stub):
    assert F.r_frag(stub, "a b", "a b c d") == pytest.approx(2.0)


def test_r_frag_below_one_when_target_is_shorter(stub):
    assert F.r_frag(stub, "a b c d", "a b") == pytest.approx(0.5)


def test_r_frag_raises_on_empty_source(stub):
    with pytest.raises(ValueError, match="empty source"):
        F.r_frag(stub, "", "a b c")


def test_fires_boundary_is_inclusive():
    assert F.fires(2.0) is True
    assert F.fires(2.001) is True
    assert F.fires(1.999) is False


def test_fires_respects_custom_threshold():
    assert F.fires(1.5, threshold=1.5) is True
    assert F.fires(1.49, threshold=1.5) is False


# --------------------------------------------------------------------------
# Positive fixtures: 5 pairs that must fire (heavy over-fragmentation)
# --------------------------------------------------------------------------

POSITIVE_FIXTURES = [
    # (source hin, over-fragmenting target, note)
    ("टिकाऊ विकास", "സുസ്ഥിര വികസനം സുസ്ഥിര വികസനം സുസ്ഥിര വികസനം", "repeated Malayalam phrase"),
    ("कंपनी ने मुनाफा कमाया।", "কোম্পানিটি লাভ করেছে এবং তারপর আরও অনেক কিছু ব্যাখ্যা করেছে যা মূল বাক্যে ছিল না।", "target much longer than source"),
    ("बाजार में वृद्धि हुई।", "మార్కెట్ పెరిగింది మార్కెట్ పెరిగింది మార్కెట్ పెరిగింది మార్కెట్ పెరిగింది", "repeated Telugu phrase"),
    ("निवेश बढ़ा।", "বিনিয়োগ বৃদ্ধি পেয়েছে এবং এটি একটি অত্যন্ত দীর্ঘ এবং অপ্রয়োজনীয় সংযোজন যা মূল পাঠ্যে ছিল না।", "long unrelated addition"),
    ("उत्सर्जन कम हुआ।", "എമിഷൻ കുറഞ്ഞു എമിഷൻ കുറഞ്ഞു എമിഷൻ കുറഞ്ഞു എമിഷൻ കുറഞ്ഞു എമിഷൻ കുറഞ്ഞു", "heavily repeated Malayalam"),
]

# --------------------------------------------------------------------------
# Negative fixtures: 5 pairs that must not fire (ordinary-length paraphrase)
# --------------------------------------------------------------------------

NEGATIVE_FIXTURES = [
    # Deliberately longer sentences (10+ words), not short ones: a very
    # short pair's ratio is coarse-grained (one extra subword swings it by
    # a large fraction), and XLM-R's vocabulary genuinely over-tokenizes
    # some Indic scripts relative to Hindi -- found while validating this
    # module against the real tokenizer, not assumed. Length gives the
    # ratio enough headroom that ordinary subword-count differences between
    # encoders do not flip the verdict.
    (
        "भारत में सतत विकास पर सरकार ने एक नई नीति की घोषणा की है।",
        "ভারতে টেকসই উন্নয়নের বিষয়ে সরকার একটি নতুন নীতির ঘোষণা করেছে।",
        "faithful same-length translation",
    ),
    (
        "कंपनी ने पिछले वित्तीय वर्ष में अपने कार्बन उत्सर्जन को उल्लेखनीय रूप से कम किया।",
        "కంపెనీ గత ఆర్థిక సంవత్సరంలో తన కార్బన్ ఉత్సర్జనాన్ని గణనీయంగా తగ్గించింది.",
        "faithful translation, longer sentence",
    ),
    (
        "बाजार में निवेशकों का भरोसा बढ़ने से शेयरों की कीमतें ऊँची हुईं।",
        "വിപണിയിൽ നിക്ഷേപകരുടെ വിശ്വാസം വർദ്ധിച്ചതിനാൽ ഓഹരി വിലകൾ ഉയർന്നു.",
        "faithful translation, longer sentence",
    ),
    (
        "सरकार ने प्रदूषण नियंत्रण के लिए सख्त नियम लागू करने का फैसला किया।",
        "সরকার দূষণ নিয়ন্ত্রণের জন্য কঠোর নিয়ম প্রয়োগ করার সিদ্ধান্ত নিয়েছে।",
        "faithful translation, longer sentence",
    ),
    (
        "अपशिष्ट प्रबंधन में सुधार लाने के लिए नगर निगम ने नई योजना शुरू की।",
        "వ్యర్థాల నిర్వహణలో మెరుగుదల తీసుకురావడానికి మున్సిపల్ కార్పొరేషన్ కొత్త ప్రణాళికను ప్రారంభించింది.",
        "faithful translation, longer sentence",
    ),
]


@pytest.mark.slow
@pytest.mark.parametrize("encoder_id", ["indicbert-v2", "xlm-r-base"])
def test_positive_fixtures_fire_on_real_tokenizers(encoder_id):
    tokenizer = get_tokenizer(encoder_id)
    for source, target, note in POSITIVE_FIXTURES:
        ratio = F.r_frag(tokenizer, source, target)
        assert F.fires(ratio), f"{encoder_id}: expected fire ({note}), ratio={ratio:.2f}"


@pytest.mark.slow
@pytest.mark.parametrize("encoder_id", ["indicbert-v2", "xlm-r-base"])
def test_negative_fixtures_do_not_fire_on_real_tokenizers(encoder_id):
    tokenizer = get_tokenizer(encoder_id)
    for source, target, note in NEGATIVE_FIXTURES:
        ratio = F.r_frag(tokenizer, source, target)
        assert not F.fires(ratio), f"{encoder_id}: expected no fire ({note}), ratio={ratio:.2f}"
