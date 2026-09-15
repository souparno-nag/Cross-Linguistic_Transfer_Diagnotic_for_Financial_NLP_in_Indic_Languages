# Capacity dilution: does the Indic-specialised encoder win? — T-505

Conneau et al. [13] describe a trade-off in multilingual pretraining: adding languages buys positive transfer between them, but spends a fixed model capacity across more of them, and past some point the dilution wins. Choosing an Indic-specialised encoder for this project was a bet on the dilution side of that trade — that a model trained on roughly a dozen Indic languages would beat one spread across a hundred.

## The answer

**The bet does not pay off here. `mbert-base` beats `indicbert-v2` in every one of the twelve comparable transfer cells and on both in-language baselines, and every one of those differences clears the two encoders' combined seed spread (T-504).** That contradicts the motivation for choosing an Indic encoder and is reported as found.

But the comparison as run **cannot** attribute that result to multilinguality, and the rest of this document is mostly about why. The two encoders differ in more than how many languages they were pretrained on, and the confound is larger than the parameter totals suggest.

## Where each encoder's capacity sits

Totals hide the thing that matters. IndicBERT-v2 is ALBERT-based and shares one layer's weights across all twelve, so most of its parameter count is the embedding table and its transformer body is very small. Counted on the `meta` device, so no weights were loaded.

| encoder | total | embeddings | transformer body | vocab | layers | shares layer params |
|---|---|---|---|---|---|---|
| indicbert-v2 | 33.4M | 25.7M | 7.8M | 200,000 | 12 | yes |
| xlm-r-base | 278.0M | 192.4M | 85.6M | 250,002 | 12 | no |
| mbert-base | 177.9M | 92.2M | 85.6M | 119,547 | 12 | no |

`indicbert-v2` has about 5× fewer parameters than `mbert-base` in total, but that understates the difference in the part that does the work. It is ALBERT-based, so one layer's weights are reused across all twelve; its transformer body is 7.8M parameters against 85.6M — **11× smaller**, not 5×. Most of its parameter count is a 200k-entry embedding table.

This is a capacity difference, not a multilinguality difference, and it runs in the same direction as the result. Any reading of the headline as "specialisation loses to breadth" has to get past it first.

## Where the Indic-specialised encoder does win: tokenisation

Mean tokens per sentence on **the same items** in four languages (task 2, block H — every arm holds the same content, §2), special tokens excluded. Fewer tokens for the same sentence means the vocabulary fits the script better, which is the concrete advantage an Indic-specialised encoder is chosen for.

| language | origin | sentences | indicbert-v2 tokens | mbert-base tokens | mbert-base ÷ indicbert-v2 | indicbert-v2 UNK/1k | mbert-base UNK/1k |
|---|---|---|---|---|---|---|---|
| hin | native | 2238 | 54.4 | 63.1 | 1.16× | 0.00 | 0.08 |
| ben | mt | 2238 | 57.4 | 70.6 | 1.23× | 0.00 | 0.00 |
| tel | mt | 2238 | 59.7 | 77.0 | 1.29× | 0.00 | 0.00 |
| mal | mt | 2238 | 69.7 | 91.9 | 1.32× | 0.00 | 0.00 |

**The specialisation advantage is real and measurable.** `indicbert-v2` needs 14% to 24% fewer tokens than `mbert-base` for the very same sentences, and the advantage *grows* as the language gets further from the centre of the corpus — smallest for Hindi, largest for Malayalam. Neither tokenizer emits a meaningful number of unknown tokens, so this is about how finely each one cuts, not about coverage failures.

So the Indic encoder does hold the advantage it was chosen for. It represents Indic text more compactly, meaning more of each sentence fits in a fixed `max_len` and each token carries more meaning. It still loses on every task measure. **A vocabulary advantage did not survive into a task advantage.**

## Where the deficit falls: in-language versus across languages

`delta` is mbert-base minus indicbert-v2; positive means mbert-base scored higher. In-language is the held-out **test** fold (T-206), not the in-language ceiling, which is the model's own training text. Cross-lingual is the mean over the computed transfer cells (T-504). `amplification` is the ratio of the two — how much bigger the deficit becomes once the model has to cross a language boundary.

| task | in-language Δ | cross-lingual Δ | cells | amplification |
|---|---|---|---|---|
| 2 | +0.0516 | +0.2678 | 6 | 5.2× |
| 3 | +0.1828 | +0.2076 | 6 | 1.1× |

**This is the most informative number in the analysis.** On task 2, `indicbert-v2` is only 0.052 macro-F1 behind in its own language, and 0.268 behind once it has to cross into another — the deficit is amplified 5.2×. An encoder that were simply too small for the task would be behind by a similar margin everywhere. This one is nearly competitive at home and collapses abroad.

The transfer gaps say the same thing from the other side: on task 2, `indicbert-v2` loses 0.434 macro-F1 crossing a language boundary against `mbert-base`'s 0.188, on the same items. Its cross-lingual scores average 0.519 on a binary task — close enough to chance that it is better described as not transferring at all than as transferring poorly.

**Task 3 does not show the amplification** (1.1×), and should not be read as contradicting it: `indicbert-v2` scores 0.183 behind in-language on a task where it only reaches 0.15 macro-F1 to begin with (532 rows over 10 classes, CLAUDE2.md T-206's data-scarcity ceiling). There is little in-language competence there to lose in transfer, so the ratio is not informative. Its task-3 gap standard deviations — ±0.17 to ±0.19 across seeds — are large enough that task-3 gap comparisons should be treated as much softer than task 2's.

## What this evidence supports, and what it does not

**Supported by measurement:**

1. `mbert-base` outperforms `indicbert-v2` on every measured cell, by +0.05 in-language and +0.21 to +0.27 cross-lingually.
2. `indicbert-v2` holds a genuine tokenisation advantage on Indic text, and it grows for languages further from the corpus centre.
3. Its deficit is concentrated in cross-lingual transfer rather than spread evenly, on the task where it is competent in-language.

**Not supported, and stated as a hypothesis only:** that the concentration in (3) reflects weaker *cross-lingual alignment* of representations rather than raw capacity. That would be the interesting reading, and points 1-3 are consistent with it, but this design cannot separate it from the 11× body-size difference. Testing it needs an encoder that varies multilinguality while holding capacity fixed.

## The comparison that would have settled it

The parameter table above contains an uncomfortable coincidence: `mbert-base` and `xlm-r-base` have **identical transformer bodies** — 85.6M parameters, same hidden size, same layer count. They differ almost entirely in vocabulary (119,547 against 250,002) and in how many languages they were pretrained on.

That pair is exactly the controlled comparison capacity dilution calls for: capacity held constant, breadth varied. It is the one arm Phase 5 could not run — VRAM-blocked (T-500): 279M parameters need 4.14 GiB of fp32 weights, gradients and AdamW state before any activation, against 3.68 GiB on this card, so it does not fit at batch size 1 — so the question T-505 was written to answer is, in its strongest form, **left open**. That is a real loss and is recorded as one rather than papered over by the two encoders that did run.

## Limitations

1. **Two encoders, not three.** XLM-R is VRAM-blocked (T-500); see above for what its absence specifically costs.
2. **The confound is unresolved.** 11× body size and a different architecture family (shared-layer ALBERT against standard BERT) move together with the multilinguality difference.
3. **Hindi-source cells only.** Twelve of eighteen transfer cells per task are blocked for both encoders because Phase 2 deferred the Bengali and Telugu baselines, so every conclusion here is about transfer *out of Hindi*.
4. **One protocol difference**, measured and stated: task 2's mBERT runs used `batch_size 8 / grad_accum 2` at an unchanged effective batch of 16 (T-502).
5. **Three seeds**, and task 3's seed spread is wide enough that its gap comparisons carry little weight.

## Bottom line

On this evidence, choosing an Indic-specialised encoder bought a better tokenizer and lost the task — but the encoder that beat it was also 11× larger where it counts, so this is not yet a demonstration that specialisation loses to breadth. The honest summary is that **the Indic-specialised encoder underperformed, and the experiment that would tell us whether multilinguality or capacity explains it did not fit on the available hardware.**

