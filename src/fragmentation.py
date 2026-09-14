"""Fragmentation ratio module (CLAUDE4.md T-602) — `tokenizer_fragmentation`.

`R_frag = |tokens(target)| / |tokens(source)|`, using the encoder's own
tokenizer, on the full sentence pair. Deliberately **not** truncated to the
model's `max_len` first: fragmentation is a property of what the tokenizer
does to the whole sentence, and truncating before measuring would hide
over-segmentation that happens past `max_len` — exactly the failure mode this
module exists to surface.

Pure text in, float out. No I/O, no torch beyond what the tokenizer itself
needs, so this module has no opinion about which encoder or which checkpoint
produced a failure — the router supplies the tokenizer (CLAUDE4.md's
encoder-agnostic hard rule).
"""

from __future__ import annotations

DEFAULT_THRESHOLD = 2.0


def token_count(tokenizer, text: str) -> int:
    """Token count with no special tokens added — a tokenizer property of the
    text alone, not of how the model would see it framed."""
    return len(tokenizer.tokenize(str(text)))


def r_frag(tokenizer, source_text: str, target_text: str) -> float:
    """`tokens(target) / tokens(source)`.

    Raises on an empty source rather than dividing by zero — corpus_io's own
    validation should never hand this module an empty source, so a zero here
    means a caller bypassed that check, not a legitimate ratio of infinity.
    """
    src_n = token_count(tokenizer, source_text)
    if src_n == 0:
        raise ValueError("empty source text — corpus_io.validate should have caught this upstream")
    return token_count(tokenizer, target_text) / src_n


def fires(ratio: float, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """`>= threshold` fires — CLAUDE4.md's own boundary, inclusive."""
    return ratio >= threshold
