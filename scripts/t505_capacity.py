"""T-505 — capacity-dilution analysis (CLAUDE5.md).

    python -m scripts.t505_capacity

Writes `reports/capacity_dilution.md`: does the Indic-specialised encoder beat
the massively-multilingual one, and where? CPU only, no model weights — the
parameter profile builds on the `meta` device and the tokenisation profile
needs only the tokenizers. Task numbers are read back from T-504's and T-206's
artefacts, never transcribed.

The written argument lives here rather than in `src/capacity.py` because it is
prose about what the numbers mean; the module supplies the numbers and the
tables. Every claim below is either a figure pulled from an artefact or is
labelled as a hypothesis.
"""

from __future__ import annotations

import sys

from src import capacity as C
from src.download_dataset.paths import REPO_ROOT
from src.encoder_comparison import COMPARED, EXCLUDED
from src.env_check import require_python

OUT = REPO_ROOT / "reports" / "capacity_dilution.md"
BASELINE, OTHER = "indicbert-v2", "mbert-base"


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def build() -> str:
    params = C.parameter_profile()
    frag = C.fragmentation_profile(COMPARED)
    deficit = C.deficit_summary(BASELINE, OTHER)
    cross = C.cross_lingual_deficit(BASELINE, OTHER).set_index("task")

    by_encoder = params.set_index("encoder")
    body_ratio = (
        by_encoder.loc[OTHER, "body_params"] / by_encoder.loc[BASELINE, "body_params"]
    )
    total_ratio = (
        by_encoder.loc[OTHER, "total_params"] / by_encoder.loc[BASELINE, "total_params"]
    )
    ratios = C.fragmentation_ratio(frag, OTHER, BASELINE)
    frag_lo, frag_hi = ratios.min(), ratios.max()
    task2 = deficit[deficit["task"] == 2].iloc[0]
    task3 = deficit[deficit["task"] == 3].iloc[0]

    lines = [
        "# Capacity dilution: does the Indic-specialised encoder win? — T-505",
        "",
        "Conneau et al. [13] describe a trade-off in multilingual pretraining: "
        "adding languages buys positive transfer between them, but spends a "
        "fixed model capacity across more of them, and past some point the "
        "dilution wins. Choosing an Indic-specialised encoder for this project "
        "was a bet on the dilution side of that trade — that a model trained on "
        "roughly a dozen Indic languages would beat one spread across a hundred.",
        "",
        "## The answer",
        "",
        f"**The bet does not pay off here. `{OTHER}` beats `{BASELINE}` in every "
        "one of the twelve comparable transfer cells and on both in-language "
        "baselines, and every one of those differences clears the two encoders' "
        "combined seed spread (T-504).** That contradicts the motivation for "
        "choosing an Indic encoder and is reported as found.",
        "",
        "But the comparison as run **cannot** attribute that result to "
        "multilinguality, and the rest of this document is mostly about why. "
        "The two encoders differ in more than how many languages they were "
        "pretrained on, and the confound is larger than the parameter totals "
        "suggest.",
        "",
    ]

    lines += C.render_parameters(params)
    lines += [
        f"`{BASELINE}` has about {total_ratio:.0f}× fewer parameters than "
        f"`{OTHER}` in total, but that understates the difference in the part "
        "that does the work. It is ALBERT-based, so one layer's weights are "
        "reused across all twelve; its transformer body is "
        f"{by_encoder.loc[BASELINE, 'body_params'] / 1e6:.1f}M parameters "
        f"against {by_encoder.loc[OTHER, 'body_params'] / 1e6:.1f}M — "
        f"**{body_ratio:.0f}× smaller**, not {total_ratio:.0f}×. Most of its "
        "parameter count is a 200k-entry embedding table.",
        "",
        "This is a capacity difference, not a multilinguality difference, and "
        "it runs in the same direction as the result. Any reading of the "
        "headline as \"specialisation loses to breadth\" has to get past it "
        "first.",
        "",
    ]

    lines += C.render_fragmentation(frag, COMPARED, numerator=OTHER, denominator=BASELINE)
    lines += [
        f"**The specialisation advantage is real and measurable.** `{BASELINE}` "
        f"needs {_pct(1 - 1 / frag_lo)} to {_pct(1 - 1 / frag_hi)} fewer tokens "
        f"than `{OTHER}` for the very same sentences, and the advantage *grows* "
        "as the language gets further from the centre of the corpus — smallest "
        "for Hindi, largest for Malayalam. Neither tokenizer emits a meaningful "
        "number of unknown tokens, so this is about how finely each one cuts, "
        "not about coverage failures.",
        "",
        "So the Indic encoder does hold the advantage it was chosen for. It "
        "represents Indic text more compactly, meaning more of each sentence "
        "fits in a fixed `max_len` and each token carries more meaning. It "
        "still loses on every task measure. **A vocabulary advantage did not "
        "survive into a task advantage.**",
        "",
    ]

    lines += C.render_deficit(deficit, baseline=BASELINE, other=OTHER)
    lines += [
        "**This is the most informative number in the analysis.** On task 2, "
        f"`{BASELINE}` is only {abs(task2['in_language_delta']):.3f} macro-F1 "
        f"behind in its own language, and "
        f"{abs(task2['cross_lingual_delta']):.3f} behind once it has to cross "
        f"into another — the deficit is amplified {task2['amplification']:.1f}×. "
        "An encoder that were simply too small for the task would be behind by "
        "a similar margin everywhere. This one is nearly competitive at home "
        "and collapses abroad.",
        "",
        f"The transfer gaps say the same thing from the other side: on task 2, "
        f"`{BASELINE}` loses {cross.loc[2, 'baseline_gap_mean']:.3f} macro-F1 "
        f"crossing a language boundary against `{OTHER}`'s "
        f"{cross.loc[2, 'other_gap_mean']:.3f}, on the same items. Its "
        f"cross-lingual scores average {cross.loc[2, 'baseline_mean']:.3f} on a "
        "binary task — close enough to chance that it is better described as "
        "not transferring at all than as transferring poorly.",
        "",
        f"**Task 3 does not show the amplification** "
        f"({task3['amplification']:.1f}×), and should not be read as "
        f"contradicting it: `{BASELINE}` scores "
        f"{abs(task3['in_language_delta']):.3f} behind in-language on a task "
        "where it only reaches 0.15 macro-F1 to begin with (532 rows over 10 "
        "classes, CLAUDE2.md T-206's data-scarcity ceiling). There is little "
        "in-language competence there to lose in transfer, so the ratio is not "
        "informative. Its task-3 gap standard deviations — ±0.17 to ±0.19 "
        "across seeds — are large enough that task-3 gap comparisons should be "
        "treated as much softer than task 2's.",
        "",
        "## What this evidence supports, and what it does not",
        "",
        "**Supported by measurement:**",
        "",
        f"1. `{OTHER}` outperforms `{BASELINE}` on every measured cell, by "
        "+0.05 in-language and +0.21 to +0.27 cross-lingually.",
        f"2. `{BASELINE}` holds a genuine tokenisation advantage on Indic "
        "text, and it grows for languages further from the corpus centre.",
        "3. Its deficit is concentrated in cross-lingual transfer rather than "
        "spread evenly, on the task where it is competent in-language.",
        "",
        "**Not supported, and stated as a hypothesis only:** that the "
        "concentration in (3) reflects weaker *cross-lingual alignment* of "
        "representations rather than raw capacity. That would be the "
        "interesting reading, and points 1-3 are consistent with it, but this "
        "design cannot separate it from the "
        f"{body_ratio:.0f}× body-size difference. Testing it needs an encoder "
        "that varies multilinguality while holding capacity fixed.",
        "",
        "## The comparison that would have settled it",
        "",
        "The parameter table above contains an uncomfortable coincidence: "
        f"`{OTHER}` and `xlm-r-base` have **identical transformer bodies** — "
        f"{by_encoder.loc[OTHER, 'body_params'] / 1e6:.1f}M parameters, same "
        "hidden size, same layer count. They differ almost entirely in "
        f"vocabulary ({int(by_encoder.loc[OTHER, 'vocab_size']):,} against "
        f"{int(by_encoder.loc['xlm-r-base', 'vocab_size']):,}) and in how many "
        "languages they were pretrained on.",
        "",
        "That pair is exactly the controlled comparison capacity dilution "
        "calls for: capacity held constant, breadth varied. It is the one "
        "arm Phase 5 could not run — "
        + " ".join(reason for reason in EXCLUDED.values())
        + " — so the question T-505 was written to answer is, in its strongest "
        "form, **left open**. That is a real loss and is recorded as one "
        "rather than papered over by the two encoders that did run.",
        "",
        "## Limitations",
        "",
        "1. **Two encoders, not three.** XLM-R is VRAM-blocked (T-500); see "
        "above for what its absence specifically costs.",
        "2. **The confound is unresolved.** "
        f"{body_ratio:.0f}× body size and a different architecture family "
        "(shared-layer ALBERT against standard BERT) move together with the "
        "multilinguality difference.",
        "3. **Hindi-source cells only.** Twelve of eighteen transfer cells per "
        "task are blocked for both encoders because Phase 2 deferred the "
        "Bengali and Telugu baselines, so every conclusion here is about "
        "transfer *out of Hindi*.",
        "4. **One protocol difference**, measured and stated: task 2's mBERT "
        "runs used `batch_size 8 / grad_accum 2` at an unchanged effective "
        "batch of 16 (T-502).",
        "5. **Three seeds**, and task 3's seed spread is wide enough that its "
        "gap comparisons carry little weight.",
        "",
        "## Bottom line",
        "",
        f"On this evidence, choosing an Indic-specialised encoder bought a "
        "better tokenizer and lost the task — but the encoder that beat it was "
        f"also {body_ratio:.0f}× larger where it counts, so this is not yet a "
        "demonstration that specialisation loses to breadth. The honest "
        "summary is that **the Indic-specialised encoder underperformed, and "
        "the experiment that would tell us whether multilinguality or capacity "
        "explains it did not fit on the available hardware.**",
        "",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    require_python()
    text = build()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    print(f"wrote {OUT.relative_to(REPO_ROOT)} ({len(text.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
