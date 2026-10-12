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

## Scope decisions

**The orthographic / numeral module is excluded from this build** (decided Oct 2026).
It is absent from `PRECEDENCE` in `src/diagnostics.py`; failures it would have caught
fall through to the next rung or to `unattributed`. The precedence list and module table
above describe the original five-module design and are kept for reference.

Why, in short (evidence and tables: `reports/digit_audit.md`, reproducible with
`python -m scripts.digit_audit`):

- **The data is not uniformly ASCII.** Hindi and Telugu natives use ASCII digits; Bengali
  natives mostly use Bengali digits (828 of 847 digit-bearing task-2 rows). MT output is
  ASCII in every language, so `ben→*` pairs change digit system on ~810–822 rows.
- **But mismatched items do not fail more often.** 12.2% / 13.4% (task 2) and 7.9% /
  10.6% (task 3) of failures carry a digit-system change, against 34% / 18% of all items
  in the same pairs. A label at precedence rung 3 would be a correlate presented as a
  cause, which hard rule 3 forbids.
- **Narrow reach.** Hindi↔Telugu conditions have no mismatch at all.
- **Currency and metric formatting were not reliably measurable**; the substring check
  tried fired inside ordinary words and was discarded.
- Limits: the evidence shows no over-representation, not that models are insensitive to
  digit script. Revisit with a controlled test. If revisited, prefer an audit-trail
  covariate over restoring the label.
- Knock-on for T-113/Phase 7: native Bengali and MT Bengali differ in digit system as
  well as provenance; any `B_nat` vs `Bn←H/T` comparison should say so.

**`morphological_masking` and `terminology_gap` are causal labels** (decided 2026-10-12).
This build is a demonstration of the diagnostic pipeline, so every module in `PRECEDENCE`
assigns a cause: a failure on which either module fires is attributed to that cause, by the
same precedence as drift and fragmentation, and Phase 7 counts and reports them as causes,
not as descriptive tags.

The validation results stay on the record, and every place the two labels are reported must
carry them as a limitation:

- **Morphological masking:** a masked term does not raise the failure rate. Masked vs intact
  term within condition, the masked item fails more often in 20 of 44 cells
  (reports/morphology_null.md).
- **Terminology gap:** salience divergence does not separate failures from correctly handled
  pairs (AUC 0.48–0.52), and the label fires at about the 5% false-positive rate its
  threshold was set to (reports/saliency_baseline.md).

In short: each label means "this module's criterion was met on this failure", which the
pipeline treats as the attributed cause. The module-level evidence that the criterion
predicts failure is, so far, absent. The orthographic module was excluded on similar
evidence; the difference is that these two are kept for the demonstration, and the choice
is stated rather than implied.

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

**Match rule fixed, 2026-10-11 (a bug, not a threshold).** The original rule
(edit distance ≤ 2 at any term length) let 2-codepoint terms match any 2-codepoint stem —
"जल"/"জল" (water) matched कर (do), हो (be), থে, and 96% of the 10,642 morphological-masking
labels of the 2026-10-09 run were such fuzzy hits (1 of 80 in the spot-check sheet was an
exact match). Ordinary inflection *does* count as masking (decided 2026-10-11), so the new
rule (`morphology.stem_matches`) accepts exactly the stem changes inflection makes: exact
match; or, for terms of ≥ 4 codepoints, the final codepoint dropped (Telugu oblique
ఉద్గారాలు → ఉద్గారాల+ను) or one vowel sign added (Bengali নির্গমনে+র). No substitution:
সম্পদ "resource" vs সম্পর্(কে) "about" was the last false positive of a looser version.
Lexicon terms are cleaned of punctuation like words are (two machine-derived entries carry a
stray `।`/`.`: ben waste, tel climate — fix them in `configs/esg_terms.json` at the next GPU
saliency run, since saliency matched against them as-is). Re-run offline on the library rung
for all four languages (`scripts.t604_remorph`): morphological masking 10,642 → 2,484 rows;
2.0–7.8% of failures per cell. Hindi targets fire 0%: Hindi writes postpositions as separate
words, so nothing attaches to the term. Known gap: multi-word terms (जीवाश्म ईंधन) can never
match a single word in this module.

**Validity check, 2026-10-12: a masked term does not make failure more likely.** The module
reads only the target sentence, so it can be run on every prediction, not just failures
(`scripts.t604_morph_null`, reports/morphology_null.md). Over every (item, seed) with a
correct source prediction, the failure rate when a lexicon term is masked vs present intact
vs absent: task 2 IndicBERT-v2 37.2% / 37.4% / 39.1%; task 2 mBERT 16.6% / 17.7% / 20.2%;
task 3 IndicBERT-v2 78.3% / 76.7% / 76.1%; task 3 mBERT 58.0% / 59.8% / 59.4%. Matched within
condition, masked fails more often than intact in only 20 of 44 cells: the module does not
yet separate failures from successes. It stays a **causal label** regardless (Scope
decisions, 2026-10-12); this result is the stated limitation that goes with it. Nothing
was tuned in response (hard rule 3).

### T-605 — Integrated Gradients module
Captum IG against a **[PAD]-embedding baseline** (revised 2026-10-06, was zero-embedding);
compare salience across the parallel pair; flag divergence.
**Done:** salience maps render for 20 sampled pairs; at most **5%** of the 40 attributions
exceed the 5% relative convergence tolerance (revised 2026-10-06, was: every one within
it). GPU-bound.

**Why the baseline changed.** Against zeros, IndicBERT-v2 failed 8/10 attributions at
50 steps and 4/10 at 200, and the worst sentence did not converge with more steps: its
error went −0.16 at 200, +0.43 at 800, +0.01 at 3200. Its logit jumps by ~0.12 inside a
~0.0005-wide band of the path near alpha 0.045, so a step grid hits or misses the jump
by luck. CPU and GPU agree exactly, so it is the model rather than numerics. Zeros lie
far off the embedding manifold. The baseline now replaces each content token with [PAD]'s
embedding and keeps the first and last positions ([CLS]/[SEP]), so only content varies
along the path. This is a method change, not a tuning step: it changes what an attribution
means, and it applies to both encoders. mBERT passed under either baseline.

**Steps are per encoder** (`saliency.N_STEPS_BY_ENCODER`): IndicBERT-v2 400, others 200.
Measured with the [PAD] baseline, 20 pairs: IndicBERT-v2 5/40 unconverged at 200, 1/40 at
400, 0/40 at 800; mBERT 0/40 at 200. Steps set only the integration's accuracy, not what
is attributed, so differing per encoder keeps the encoders comparable.

**Why a rate and not "every one".** The residual is a property of the model's path, not of
the step count. Requiring zero would mean raising steps until a fixed sample happens to
pass. Unconverged rows are still labelled `not_converged` and never fire. The rate gates
the method, and those rows stay visible. 20 pairs is the smallest sample in which 5% is
resolvable; with 5 pairs it would mean zero.

Do not loosen the 5% tolerance or the 5% rate to make the check pass (hard rule 3).
Raise the steps or report the failure.

**Follow-up, 2026-10-09 run.** The full run missed the 5% tolerance in two cells (task 3
IndicBERT-v2 17.9% unconverged at 400 steps, task 2 mBERT 14.3% at 200; the other two 3.8%
and 3.1%). Those two cells move to 800 steps (`saliency.N_STEPS_BY_TASK_ENCODER`), and only
their unconverged rows are recomputed (`scripts.t605_resaliency`): a converged row is already
accurate to within tolerance. Each audit row now records `saliency_n_steps`.

**Divergence threshold: set against a null, not against the failures.** The 0.3 placeholder
never fires (failures' 95th percentile +0.04 to +0.10). Reading a value off the failures'
own distribution would fix the firing rate by construction, which hard rule 3 forbids.
`scripts.t605_baseline` instead measures divergence on pairs the model got *right* in both
languages; the candidate threshold is that null's 95th percentile (a ~5% false-positive rate
fixed in advance). If failures exceed it at about the null's own rate, the module detects
nothing beyond chance, and that is reported as the result. The chosen value is applied with
`scripts.t601_rethreshold` and recorded here.

**Threshold set, 2026-10-11.** Per cell, at the null's 95th percentile
(`saliency.DIVERGENCE_THRESHOLD_BY_TASK_ENCODER`; reports/saliency_baseline.md): task 2
IndicBERT-v2 0.038, task 2 mBERT 0.047, task 3 IndicBERT-v2 0.085, task 3 mBERT 0.078;
fallback 0.065 (pooled null). Per cell because the null's spread differs by cell, and one
value would hold each cell to a different false-positive rate. **Result: divergence does not
separate failures from correct pairs** — failure-vs-null AUC 0.48–0.52, and failures exceed
the thresholds at 3.7–8.7% (bootstrap 95% CIs mostly spanning the null's own 5%).
`terminology_gap` now fires on 1.1–2.2% of failures, at about the chance rate. It stays a
**causal label** regardless (Scope decisions, 2026-10-12); this result is the stated
limitation that goes with it.
Convergence after the 800-step re-run: task 2 mBERT 14.3% → 2.1% unconverged; task 3
IndicBERT-v2 17.9% → 8.2%, still above the 5% limit and reported as such.

**Re-checked against the GPU run logs, 2026-10-12; thresholds unchanged.** The baseline
completed for all four cells (the 2026-10-09 attempt died on a CUDA launch failure and was
re-run on 2026-10-10). Recomputed from the saved rows: each set value equals its cell's null
q95 (bootstrap 95% CIs: 0.032–0.052, 0.038–0.057, 0.066–0.094, 0.066–0.093); AUC 0.50 / 0.52 /
0.52 / 0.48, condition-matched 0.49 / 0.52 / 0.52 / 0.47. Task 3's null rests on only 75–78
distinct items (few headlines carry a lexicon term in both languages), so its threshold is
the least precise. The transformers "incorrect regex pattern … fix_mistral_regex" warning
in the logs is cosmetic in 4.57.6: no pre-tokenizer change is applied unless the flag is
passed, so tokenization matches training.

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
