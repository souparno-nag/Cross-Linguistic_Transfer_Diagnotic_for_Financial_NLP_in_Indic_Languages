"""Integrated Gradients module (CLAUDE4.md T-605, this build's T-604) —
`terminology_gap`.

Compares per-token salience (via Captum's Integrated Gradients against a
zero-embedding baseline) between a source sentence and its translated
target, for the subset of instances whose source sentence contains a
curated ESG lexicon term (`configs/esg_terms.json`). If the term's own
tokens carry a much smaller share of total salience on the target side than
on the source side, the model is attending to the term less in translation —
a terminology gap, as distinct from a fluency or morphology problem.

CLAUDE4.md's "Eq. 6" is not spelled out anywhere in that document; the
divergence metric below (`term_salience_share` difference, fired at a
threshold set once from real inspection) is a metric designed for this
module, not one transcribed from a source, and is documented as such.

**Coverage caveat, reported rather than hidden**: only instances whose
*source* sentence contains one of the curated lexicon terms are evaluated at
all. Everything else is `not_applicable` — distinct from `not_fired`
(evaluated, below threshold) and `unavailable` (an infra failure) — and
falls straight through to `unattributed`. No cross-lingual word-alignment
tool exists in this codebase, so a term-presence check on the source side is
the only honest way to decide what this module can even ask about.

The pure arithmetic (`term_token_span_from_offsets`, `salience_share`,
`fires`) needs neither torch nor Captum, so it can be unit-tested on
synthetic salience arrays without a GPU. `compute_salience` and
`diagnose_instance`, which actually run the model, import both lazily.
"""

from __future__ import annotations

from dataclasses import dataclass

DEFAULT_N_STEPS = 50
# Set once from inspecting real task_2 examples during T-604's build, and
# documented rather than re-tuned to tidy the resulting distribution
# (CLAUDE4.md hard rule 3). Until that inspection happens this is a
# placeholder carried over from the approved plan, not yet a measured value.
DEFAULT_DIVERGENCE_THRESHOLD = 0.3
# Integrated Gradients' completeness axiom says the attributions sum to
# F(x) - F(baseline); Captum's convergence delta is the error in that
# identity. It is judged *relative* to |F(x) - F(baseline)|, because an
# absolute delta means nothing without the scale of the logit it is an error
# in. 5% is a conventional judgement, not a derived figure; an unconverged
# attribution is reported as `not_converged` and never fires (CLAUDE4.md T-605).
DEFAULT_CONVERGENCE_TOL = 0.05


@dataclass
class SaliencyResult:
    # "fired" | "not_fired" | "not_applicable" | "unavailable" | "not_converged"
    status: str
    divergence: float | None = None
    convergence_delta: float | None = None  # relative; see DEFAULT_CONVERGENCE_TOL
    concept_id: str | None = None


def find_lexicon_term(text: str, concepts: list[dict], lang: str) -> tuple[str, dict] | None:
    """The first lexicon concept whose surface form for `lang` occurs as a
    substring of `text`, or `None` if none of them do.

    A concept missing a surface form for `lang` (still `null` before
    `scripts.t603_esg_terms` runs) is skipped, never treated as a match.
    """
    for concept in concepts:
        term = concept.get(lang)
        if term and term in text:
            return term, concept
    return None


def term_token_span_from_offsets(
    text: str, term: str, offset_mapping: list[tuple[int, int]]
) -> list[int] | None:
    """Token indices (into `offset_mapping`) covering `term`'s first
    occurrence in `text`, or `None` if it cannot be found in either."""
    start = text.find(term)
    if start == -1:
        return None
    end = start + len(term)
    span = [
        i
        for i, (tok_start, tok_end) in enumerate(offset_mapping)
        if tok_end > start and tok_start < end and not (tok_start == 0 and tok_end == 0)
    ]
    return span or None


def salience_share(salience, span: list[int]) -> float:
    """Fraction of total salience covered by `span`'s tokens. `0.0` for a
    degenerate all-zero salience array rather than a division error."""
    total = float(sum(salience))
    if total <= 0:
        return 0.0
    return float(sum(salience[i] for i in span)) / total


def converged(relative_delta: float, tol: float = DEFAULT_CONVERGENCE_TOL) -> bool:
    """`<= tol` converged. NaN is not converged (`nan <= tol` is False)."""
    return relative_delta <= tol


def fires(divergence: float, threshold: float = DEFAULT_DIVERGENCE_THRESHOLD) -> bool:
    return divergence >= threshold


def compute_salience(
    loaded,
    tokenizer,
    text: str,
    gold_label: int,
    *,
    max_len: int,
    n_steps: int = DEFAULT_N_STEPS,
    device: str = "cpu",
):
    """Per-token salience (L2 norm of the IG attribution) and the **relative**
    convergence error `|delta| / |F(x) - F(baseline)|` (Captum's own delta,
    scaled by the logit gap it should account for), for one sentence against a
    zero-embedding baseline.

    `embed_layer.get_input_embeddings()` is HF's architecture-agnostic
    accessor — it works identically whether the encoder is ALBERT-based
    (IndicBERT-v2), RoBERTa-based (XLM-R) or BERT-based (mBERT), which is
    what keeps this module encoder-agnostic without per-architecture
    branching (CLAUDE4.md hard rule 6).
    """
    import torch  # noqa: PLC0415
    from captum.attr import IntegratedGradients  # noqa: PLC0415

    model = loaded.model
    encoding = tokenizer(
        text, return_tensors="pt", return_offsets_mapping=True, truncation=True, max_length=max_len
    )
    offset_mapping = encoding.pop("offset_mapping")[0].tolist()
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    embed_layer = model.encoder.get_input_embeddings()
    input_embeds = embed_layer(input_ids)
    baseline = torch.zeros_like(input_embeds)

    def forward_fn(embeds, mask):
        out = model.encoder(inputs_embeds=embeds, attention_mask=mask)
        return model.classifier(out.last_hidden_state[:, 0])

    ig = IntegratedGradients(forward_fn)
    attributions, delta = ig.attribute(
        input_embeds,
        baselines=baseline,
        additional_forward_args=(attention_mask,),
        target=int(gold_label),
        n_steps=n_steps,
        return_convergence_delta=True,
    )
    with torch.no_grad():
        gap = (
            forward_fn(input_embeds, attention_mask)[0, int(gold_label)]
            - forward_fn(baseline, attention_mask)[0, int(gold_label)]
        ).abs().item()
    # a vanishing gap makes the ratio meaningless; report it as unconverged
    # (inf) rather than dividing by ~0 and calling the result good.
    relative = float(delta.abs().max().item()) / gap if gap > 1e-6 else float("inf")
    salience = attributions.norm(dim=-1).squeeze(0).detach().cpu().tolist()
    return salience, offset_mapping, relative


def diagnose_instance(
    loaded,
    tokenizer,
    source_text: str,
    target_text: str,
    gold_label: int,
    concepts: list[dict],
    lang_src: str,
    lang_tgt: str,
    *,
    max_len: int,
    threshold: float = DEFAULT_DIVERGENCE_THRESHOLD,
    n_steps: int = DEFAULT_N_STEPS,
    convergence_tol: float = DEFAULT_CONVERGENCE_TOL,
    device: str = "cpu",
) -> SaliencyResult:
    """One instance's saliency-divergence verdict, or `not_applicable` if the
    source sentence carries none of the lexicon's terms, or the term's
    target-language surface form is missing, or either occurrence's token
    span cannot be recovered."""
    found = find_lexicon_term(source_text, concepts, lang_src)
    if found is None:
        return SaliencyResult(status="not_applicable")
    _, concept = found
    src_term = concept[lang_src]
    tgt_term = concept.get(lang_tgt)
    if not tgt_term:
        return SaliencyResult(status="not_applicable", concept_id=concept["id"])

    src_salience, src_offsets, src_delta = compute_salience(
        loaded, tokenizer, source_text, gold_label, max_len=max_len, n_steps=n_steps, device=device
    )
    tgt_salience, tgt_offsets, tgt_delta = compute_salience(
        loaded, tokenizer, target_text, gold_label, max_len=max_len, n_steps=n_steps, device=device
    )

    src_span = term_token_span_from_offsets(source_text, src_term, src_offsets)
    tgt_span = term_token_span_from_offsets(target_text, tgt_term, tgt_offsets)
    if src_span is None or tgt_span is None:
        return SaliencyResult(status="not_applicable", concept_id=concept["id"])

    src_share = salience_share(src_salience, src_span)
    tgt_share = salience_share(tgt_salience, tgt_span)
    divergence = src_share - tgt_share
    worst_delta = max(src_delta, tgt_delta)
    if not converged(worst_delta, convergence_tol):
        # kept for the audit trail, but an attribution that fails its own
        # completeness check cannot support a verdict either way
        status = "not_converged"
    else:
        status = "fired" if fires(divergence, threshold) else "not_fired"
    return SaliencyResult(
        status=status,
        divergence=divergence,
        convergence_delta=worst_delta,
        concept_id=concept["id"],
    )
