# CLAUDE.md — Phase 6: Diagnostic Pipeline (C3)

Scope: **Phase 6 only.** Build the failure router that turns a wrong prediction into a
named linguistic cause. Do **not** aggregate results into findings (Phase 7) or run new
encoders (Phase 5). If a request drifts there, say so and stop.

---

## Running before Phase 5 — read this first

Phase 6 is being built ahead of Phase 5 (encoder extension). This is the right call:
C3 is the project's distinguishing contribution and has no schedule slack, while a
three-encoder comparison is deferrable and IndicBERT-v2 alone supports a complete
result.

One consequence, and it is the main design constraint on this phase:

> **Every module must be tokenizer- and encoder-agnostic.** The fragmentation ratio and
> Integrated Gradients modules both depend on the encoder. Parameterise them by
> `encoder_id`; never hardcode IndicBERT-v2. When Phase 5 lands, re-running the pipeline
> on XLM-R and mBERT must cost a config change, not a rewrite.

Store `encoder_id` on every diagnostic output row from day one.

---

## Depends on

- `data/failures/{condition}.parquet` — Phase 3, T-307
- `data/predictions/{condition}.parquet` — Phase 3, T-302
- `src/labse_gate.py` — Phase 1, T-108. **Reuse it.** Do not reimplement the gate.
- `data/v1.0/` — frozen corpus, read-only

---

## Architecture

Two stages, strictly ordered.

**Stage 1 — semantic gate.** LaBSE cosine similarity below τ routes the instance to
`translation_drift` and it goes no further.

**Stage 2 — linguistic attribution.** Only semantically equivalent pairs reach the four
linguistic modules.

The ordering is the whole point. If the linguistic modules ran first, noise from your own
machine translation would be misattributed to tokenization or morphology and the
diagnosis would be unsound. Never reorder these stages.

### Resolution precedence

An instance may satisfy several modules. Exactly one label is assigned, by this
precedence:

```
1. translation_drift          (Stage 1 — terminal)
2. tokenizer_fragmentation
3. orthographic_numeral_mismatch
4. morphological_masking
5. terminology_gap            (saliency divergence)
6. unattributed               (nothing fired)
```

`unattributed` is a legitimate outcome and must be reported, not hidden. A pipeline that
labels 100% of failures is suspicious.

---

## Modules

| Module | Fires when | Label |
|---|---|---|
| LaBSE gate (Eq. 5) | `sim < τ` (τ from Phase 1 T-110) | `translation_drift` |
| Fragmentation ratio (Eq. 4) | `R_frag ≥ 2.0` | `tokenizer_fragmentation` |
| Orthographic / numeral | digit-system or currency mismatch | `orthographic_numeral_mismatch` |
| Morphological masking | bound morphology masks the root financial noun | `morphological_masking` |
| Integrated Gradients (Eq. 6) | salience diverges across the parallel pair | `terminology_gap` |

`R_frag = |tokens(target)| / |tokens(source)|`, using **the encoder's own tokenizer**.

---

## Hard rules

1. **Exactly one label per instance**, plus a full audit trail of every module that
   fired. A count is not a diagnosis.
2. **Never reorder the stages.** Gate first, always.
3. **Never tune a threshold to improve the distribution.** Thresholds are set on
   principle and evidence, then reported. Tuning them to produce a tidy result is
   fabrication.
4. **Never drop an instance.** `unattributed` exists for this.
5. **Deterministic.** Same input + same config → same labels.
6. **Encoder-agnostic.** See the Phase 5 note above.
7. **No aggregation, no findings.** That is Phase 7. This phase produces labels.

---

## Tasks

### T-601 — Pipeline scaffold
Two-stage router, precedence resolution, audit log.
**Done:** end-to-end run on 10 fixtures produces exactly one label each plus a trace of
all modules that fired.

### T-602 — Fragmentation ratio module
Tokenize source and target with the encoder's tokenizer, compute `R_frag`, fire at ≥ 2.0.
**Done:** hand-verified on 10 known-fragmenting examples; correct for at least two
different tokenizers.

### T-603 — Orthographic / numeral module
Deterministic regex over digit systems, currency symbols and metric formatting. Digit
ranges in §Reference below.
**Done:** unit tests cover all five digit systems and mixed-script cases.

### T-604 — Morphological masking module ⚠️ SPIKE FIRST
**Stanza has no Malayalam model** — there is no Malayalam treebank in Universal
Dependencies. Confirm this before writing the module, and check Telugu availability too.

Fallback ladder, in order:
1. Indic NLP Library morphological analyser
2. Rule-based suffix stripping from a curated case-marker / postposition list per language
3. Declare the module unavailable for that language and report it as unmeasured

**Done:** module runs for every target language, or the gap is documented and the
affected instances fall through to the next precedence level. **Decide by 25 Sep.**

### T-605 — Integrated Gradients module
Captum IG against a zero-embedding baseline; compare salience across the parallel pair;
flag divergence.
**Done:** salience maps render for 5 sampled pairs; convergence delta checked and within
tolerance. GPU-bound.

### T-606 — Multi-fire audit trail
Record every module that fired per instance, not just the winner.
**Done:** any assigned label traces to its deciding module and to what else matched.

### T-607 — Module unit tests
Purpose-built fixtures that trigger each module in isolation.
**Done:** ≥5 positive and ≥5 negative fixtures per module, passing.

### T-608 — Manual spot-check
50 sampled failures per target language, judged by hand against the module's verdict.
**Done:** agreement rate reported; ≥80% target; systematic disagreements fed back into
thresholds **once**, with the change documented — not iterated until the numbers look
good.

---

## Outputs

| Artefact | Path |
|---|---|
| Per-instance labels | `data/diagnostics/{condition}.parquet` |
| Audit trail | `data/diagnostics/audit/{condition}.parquet` |
| Spot-check results | `reports/spotcheck.md` |

Label row columns: `condition_id, encoder_id, item_id, block_id, src_lang, tgt_lang,
assigned_label, modules_fired, r_frag, labse_sim, seed`.

---

## Reference — digit systems

| Script | Digits | Codepoints |
|---|---|---|
| Bengali | `০-৯` | U+09E6–U+09EF |
| Devanagari | `०-९` | U+0966–U+096F |
| Telugu | `౦-౯` | U+0C66–U+0C6F |
| Malayalam | `൦-൯` | U+0D66–U+0D6F |
| ASCII | `0-9` | U+0030–U+0039 |

---

## Working agreement

- **Report the unattributed rate prominently.** It is the honest measure of the
  pipeline's coverage.
- **If a module cannot run for a language, say so.** Silently skipping it produces a
  distribution that looks complete and is not.
- **Do not interpret the labels.** Naming which cause dominates for which language pair
  is Phase 7 and needs the aggregate view.
- Prefer boring code. This must be reproducible cold in October.
