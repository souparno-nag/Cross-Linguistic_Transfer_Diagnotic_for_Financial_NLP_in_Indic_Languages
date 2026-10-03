# III. Implementation

## A. System Overview

The system is organized into six phases, of which the first five are complete and
the sixth is under construction. Table I summarizes phase status as of this writing.
Phases 1–3 were built in numerical order; for scheduling reasons, work on Phase 6
(diagnostics) was started early and then paused pending the completion of Phase 5
(encoder extension), after which Phase 4 (source-language comparison) was completed.
The dependency structure, not the build order, is presented below.

**TABLE I. PHASE STATUS**

| Phase | Description | Status |
|---|---|---|
| 1 | Parallel corpus construction | Complete |
| 2 | Training pipeline and in-language baselines | Complete |
| 3 | Zero-shot transfer evaluation | Complete (Hindi source) |
| 4 | Source-language comparison (Bengali, Telugu) | Complete |
| 5 | Encoder extension (mBERT; XLM-R deferred) | Complete |
| 6 | Diagnostic pipeline (failure attribution) | Scaffolded, not run at scale |

Every phase reads and writes through a small number of shared, single-purpose
modules rather than duplicating logic: `src/corpus_io.py` is the sole reader/writer
of corpus Parquet files, `src/ids.py` owns key construction, `src/labse_gate.py` is
reused unmodified by both the Phase 1 quality gate and the Phase 6 diagnostic gate,
and `src/data.py`, `src/config.py`, and `src/metrics.py` are shared by every training
and evaluation script from Phase 2 onward. All artifacts are content-addressed by a
logged configuration hash, and every script that touches the GPU is written to be
resumable, reflecting the project's development environment: a single 4 GB laptop
GPU (RTX 3050) with intermittent availability.

## B. Phase 1 — Parallel Corpus Construction

**Source data.** The corpus is derived from IndicFinNLP (Ghosh et al., LREC-COLING
2024), obtained via Kaggle and committed to the repository with per-file SHA-256
checksums. It supplies three financial-NLP tasks — numeral-span annotation
(`task_1`), binary sustainability classification (`task_2`), and 10-class ESG-topic
classification (`task_3`) — natively in Hindi, Bengali, and Telugu only.

**Independence audit.** Before any translation was generated, an audit (`T-102`)
tested whether the three native language splits of each task were independently
authored or were the same content presented in different languages. Three tests were
used: a structural check on class proportions, an order-aligned cosine check, and a
nearest-neighbor cosine check over the full target split using LaBSE embeddings,
calibrated against positive and negative controls. The audit found that `task_1`'s
splits are independently sourced (median cross-language nearest-neighbor similarity
0.58–0.64), while `task_2` and `task_3` are the same underlying sentences,
professionally human-translated and released in shuffled row order (median 0.88–0.91,
and confirmed outright for `task_3` via its shared `URL` column). This finding
determined the rest of the phase's design: `task_2` and `task_3` required an
item-alignment step (`T-102b`) — mutual nearest-neighbor matching for `task_2`,
exact `URL` join for `task_3` — before cross-language correspondence could be
established, while `task_1` did not.

**Translation.** Each native split was machine-translated into the remaining three
languages using `ai4bharat/indictrans2-indic-indic-1B`, invoked directly
Indic→Indic with no English pivot, under a single frozen decoding configuration
(beam size 5, max length 256, no sampling) applied identically across all nine
translation directions. The 1B parameter model was selected over a 320M distilled
alternative despite an 8× speed penalty, because output-corruption analysis (`T-105`)
showed the smaller model corrupted more than twice as many rows (10.0% vs. 4.4% on
`task_1`) — a quality difference invisible to a naive digit-preservation proxy.
Generation covers all nine directions across all three tasks: 92,760 machine-translated
rows in total (68,358 for `task_1`, 19,614 for `task_2`, 4,788 for `task_3`), with
identical-sentence deduplication cutting `task_1`'s translation volume by 41% before
generation. All translation jobs checkpoint per batch and resume from `item_id`,
required by the intermittent GPU budget.

**Quality control.** Three complementary automated checks were implemented, each
catching a different failure mode that the others cannot: (1) a scale- and
script-aware numeral/entity checker (`src/entities.py`, `T-105`) that compares
financial quantities by value rather than by digit string, so that equivalent
expressions (e.g., a scale-word rewrite from "100 million" to "10 crore") are not
misreported as numeral loss; (2) a structural-integrity checker (`src/integrity.py`,
`T-107`) that detects row-alignment failure, entity-placeholder leakage, and a
nukta-driven Unicode-escape leakage mode specific to this MT pipeline; and (3) a
LaBSE cosine-similarity gate (`src/labse_gate.py`, `T-108`) computed over every one
of the 92,760 machine-translated pairs, with the acceptance threshold τ calibrated
per task rather than fixed globally (`T-110`), because cosine similarity was found
to fall with text length independent of translation quality. No row is ever dropped
for low similarity or detected corruption; every check attaches a flag to the row
(`translation_drift`, `entity_loss`, `placeholder_leak`, `escape_leak`,
`span_not_recovered`, etc.) and the row is retained, per the project's standing rule
that below-threshold pairs are a reported research category, not discarded data.

**Task-1-specific handling.** Because `task_1` annotates a numeral by character
offset, and a translated sentence differs in length and script from its source, the
offsets cannot be carried across translation. A dedicated span-recovery module
(`src/spans.py`, part of `T-106`) re-locates the annotated numeral in the translated
output by value rather than by position, distinguishing a genuinely lost numeral from
one that was only rescaled or listed alongside others, and records the outcome
per row (`span_recovered`).

**Freeze and release.** Once each task's checks passed, its twelve splits (one
native and eight machine-translated arms per task, spanning three source blocks) were
copied into an immutable, checksummed release directory (`data/v1.0/`, `T-112`), and
an automatically generated datasheet (`data/v1.0/DATASHEET.md`, `T-114`) was produced
directly from the frozen artifacts rather than hand-transcribed, so that it cannot
silently drift from the data it describes. A companion evaluation-condition matrix
(`configs/eval_conditions.json`, `T-113`) enumerates every valid train/evaluate pairing
and is validated at two levels — no split is both a training source and an evaluation
target, and, for `task_2`/`task_3`, no training item and evaluation item share the
same underlying content after alignment — closing the row-shuffling risk identified
by the `T-102` audit.

## C. Phase 2 — Training Pipeline and In-Language Baselines

Phase 2 implements a deliberately minimal fine-tuning architecture: a pretrained
encoder's `[CLS]` representation followed by a single linear classification head,
trained with cross-entropy loss. The model registry (`src/data.py`) supports
IndicBERT-v2, mBERT, and XLM-R uniformly, though only IndicBERT-v2 was trained in
this phase. Every configuration is expressed as a hashed YAML file (`src/config.py`)
and trained for a minimum of three random seeds, reported as mean ± standard
deviation rather than as a single run. Training is restricted by hard rule to the
native (human-written) splits only, since training on machine-translated data would
contaminate every downstream transfer measurement.

**Validation gate.** Because an unvalidated baseline would make any later transfer
result uninterpretable — there would be no way to separate "the model fails to
transfer" from "the model was never trained correctly" — Phase 3 is gated on Phase 2
reproducing IndicFinNLP's own published in-language numbers within a ±0.02 tolerance
(`src/baseline_gate.py`, `T-207`). Results and gate status are reported in Section
IV-B.

## D. Phase 3 — Zero-Shot Transfer Evaluation

Phase 3 evaluates the frozen Hindi-trained checkpoints from Phase 2 on the other
languages with no further parameter updates (`src/inference.py`, `T-301`, asserted to
take no gradient step). For every evaluation condition drawn from Phase 1's condition
matrix, it logs a per-instance prediction record (`item_id`, gold label, predicted
label, class probabilities; `T-302`), computes accuracy, macro-F1, per-class F1 and a
confusion matrix (`T-303`), and reports a bootstrap confidence interval together with
the **transfer gap** — source (in-language) score minus target (other-language) score,
computed on the same underlying item wherever alignment permits (`T-304`). A separate
module (`T-306`) flags "mismatches": instances correct in the source language but
incorrect in the target, and a companion module (`T-307`) exports the resulting
failed-instance set as the explicit input contract for the Phase 6 diagnostic
pipeline. All (condition × seed) combinations defined by the condition matrix are run
in one sweep (`T-305`) and rendered into the final results tables (`T-308`).

## E. Phase 5 — Encoder Extension

Phase 5 repeats the Phase 2/3 protocol on a second pretrained encoder, mBERT-base, to
test whether IndicBERT-v2's Indic-specific pretraining was in fact the right encoder
choice. A third candidate, XLM-R-base, was evaluated for feasibility and found
unable to fit on the project's 4 GB GPU: under mixed-precision fine-tuning, fixed
memory cost — full-precision weights, gradients, and two AdamW moment buffers per
parameter, 16 bytes/parameter total — exceeds available headroom before a single
training sentence is loaded (4.14 GiB required against 3.68 GiB available). This cost
is independent of batch size or sequence length, so it cannot be worked around by
tuning; XLM-R's configuration is retained under `configs/train/deferred/` for a future
run on larger hardware, and an adapter-based workaround was evaluated and rejected
(`T-503`) because it would confound the pretraining-only comparison the phase exists
to make. A tokenizer-level fragmentation comparison between IndicBERT-v2 and mBERT was
implemented as part of this phase's analysis (`src/fragmentation.py` origin) and was
later reused directly by the Phase 6 scaffold.

## F. Phase 4 — Source-Language Comparison

Phase 4 extends the training and evaluation pipeline, unchanged in code, to two
additional source languages, Bengali and Telugu, on both encoders that fit the
available hardware. The phase brief predicted that this would be a configuration
change rather than a code change; that held, with the addition of a new consistency
guard that compares each new source-language configuration against the
same-encoder Hindi configuration (rather than against another encoder's
configuration, which Phase 5's existing guard already checked), to prevent a
training-recipe difference from being misattributed to a language difference. This
phase completes the full reachable grid: 63 of 63 (condition × seed) pairs on each
task, for each of the two encoders — 126 results per task, up from 21 of 63 before
this phase.

## G. Phase 6 — Diagnostic Pipeline (In Progress)

Phase 6 is the project's not-yet-complete component. Its purpose is to take each
transfer failure identified in Phases 3/4 (`data/failures/{condition}.parquet`) and
assign it exactly one linguistic cause, so that the *size* of the transfer gap
measured so far can eventually be explained rather than only reported. Its
architecture is fixed by specification (`CLAUDE4.md`) as a strictly ordered two-stage
router: a semantic gate first (reusing the Phase 1 LaBSE-similarity module
unmodified, so that translation noise cannot be misattributed to a linguistic
cause), followed — only for pairs that pass the gate — by a set of linguistic
attribution modules, with a fixed resolution precedence so that every instance
receives exactly one label plus a full audit trail of every module that fired.

As of this writing, the following has been implemented and unit-tested:

- The two-stage router and precedence-resolution scaffold (`src/diagnostics.py`,
  `T-601`), which reuses `src.labse_gate.score_items` for the gate rather than
  reimplementing it, and produces both a per-instance label file and a full
  per-module audit trail (`T-606`).
- A tokenizer-fragmentation module (`src/fragmentation.py`, `T-602`), computing the
  ratio of target-to-source token counts under the encoder's own tokenizer and
  firing at a fixed threshold.
- A morphological-masking module (`src/morphology.py`, `T-604`) with a documented
  fallback ladder, needed because no Universal Dependencies treebank exists for
  Malayalam and none was assumed to exist for Telugu without being checked first.
- A saliency-divergence module (`src/saliency.py`, `T-605`) using Integrated
  Gradients against a zero-embedding baseline to compare token-level attribution
  between a source item and its target-language counterpart.
- A machine-translated ESG term lexicon (`configs/esg_terms.json`, `T-603`),
  hand-curated in Hindi and translated into Bengali, Telugu, and Malayalam via the
  Phase 1 translation model, feeding the terminology-gap signal.
- A manual spot-check harness (`scripts/t607_spotcheck.py`, `T-607`), not yet
  exercised.

Two things remain open and constitute the bulk of the pending work:

1. **The orthographic/numeral-mismatch module** from the original five-module design
   is deliberately excluded and is absent from the live precedence chain; failures it
   would have caught fall through to `unattributed` or another module. This was checked
   rather than assumed: Bengali natives use Bengali digits while MT output is ASCII, but
   mismatched items fail at or below the base rate, so a label would be spurious
   (CLAUDE4.md "Scope decisions"; `reports/digit_audit.md`).
2. **The pipeline has been exercised on one condition only** — `task_2`,
   Hindi→Bengali, both the native- and machine-translated-target arms
   (`data/diagnostics/task_2/transfer_hin_to_ben*.parquet`) — as an end-to-end
   correctness check. It has not yet been run over the full failure-set grid produced
   by Phases 3 and 4 (both tasks, all source/target/encoder combinations), no manual
   spot-check against human judgment has been performed (`T-608`), and no
   `reports/spotcheck.md` or aggregate diagnostic report yet exists. Running the
   pipeline at scale, spot-checking its labels, and — as a strictly separate,
   not-yet-started phase (Phase 7) — aggregating the resulting labels into
   root-cause findings, is the immediate next work.

### G.1 Phase 6 test status (as of 3 Oct 2026)

- **Fast tests (`-m "not slow"`, CPU only, no model download): 68 passed** across
  `test_diagnostics`, `test_fragmentation`, `test_morphology`, `test_saliency`
  (the saliency set includes a toy-model run of the real Captum path).
- **Not run (9 slow tests):** real-tokenizer fragmentation on IndicBERT-v2 and XLM-R
  (IndicBERT-v2 is gated and needs an HF token), the real Indic NLP morphology library
  for all four languages, and the five-pair Integrated Gradients render. These need
  downloads and, for the last, a GPU. Treat T-602/T-604/T-605 as unverified on real
  encoders until they pass.
- **Full suite, same flags:** 533 passed; 19 failed and 21 errored in other phases'
  tests, all traced to the scratch environment used (Python 3.13 rather than the required
  3.11; `tabulate`, `openpyxl` missing; HuggingFace offline), none in Phase 6 code. Not
  re-run in the project's own environment.
- **T-605 convergence check:** implemented. `compute_salience` returns
  `|delta| / |F(x) - F(baseline)|`; above 5% (`DEFAULT_CONVERGENCE_TOL`, a conventional
  judgement) the instance is `not_converged`, never fires, and the error is recorded in
  the audit as `saliency_convergence_error`.
- **Still open:** the divergence threshold (0.3) is a placeholder and has not been
  inspected on real salience distributions; T-608 spot-check not filled in.
- **Observation for the spot-check, not a conclusion:** on the one diagnosed condition
  (task 2, Hindi to Bengali) morphological masking fires on 62% of failures (678/1094
  and 849/1423), while fragmentation and terminology-gap fire on none. That is the
  pattern a loose fuzzy match would produce, so the spot-check should look at it
  specifically before any other condition is run.
- **Other conditions:** `scripts/t601_diagnostics.py` now covers all tasks, source blocks
  and encoders (resumable, no silent CPU fallback, encoder-namespaced output). Not yet
  run on anything but Hindi to Bengali.

## H. Engineering Constraints Common to All Phases

Several constraints recur across every phase and shaped implementation choices
throughout. All corpus data is stored as Parquet, never CSV, because Indic scripts
and financial numerals break under CSV quoting rules; every script accepts an
explicit random seed and produces byte-identical output across runs with the same
configuration; every artifact is logged against the configuration hash that produced
it; and every long-running GPU job checkpoints per batch and resumes by key, a
requirement driven by the single 4 GB GPU with intermittent availability that
development was done on. Several silent failure modes were found and corrected
during implementation, notably an OOM-inducing memory-allocator fragmentation issue
requiring `expandable_segments`, an IndicTrans2 tokenizer bug that surfaces only on
`transformers` ≥5 and is not shimmable, and a checkpoint-resume bug in which resumed
translation rows read back from Parquet as a numpy array rather than a list and were
consequently — and incorrectly — flagged as empty output while their translations
were in fact intact and correct.

---

# IV. Results

## A. Corpus Quality (Phase 1)

Calibration of the LaBSE acceptance threshold τ against known-good human-translation
pairs (`T-110`) found that a single global τ = 0.82 is not appropriate across tasks:
it rejects 0.0% of human-translated pairs on `task_2` (median text length 148
characters) but 23.4% on `task_3` (median 86 characters), because cosine similarity
falls with shorter context independent of translation quality. τ was therefore
calibrated per task (Table II); `task_1` has no human reference in any direction and
its threshold remains inherited rather than calibrated. A further finding, used
throughout the interpretation of downstream results, is that machine translations
scored **higher** than human translations of the same items on the LaBSE measure in
every comparable case — evidence that the metric rewards literalness rather than
adequacy, and should be read as a relative ranking tool rather than an absolute
quality score.

**TABLE II. CALIBRATED SIMILARITY THRESHOLD BY TASK**

| Task | τ | Basis |
|---|---|---|
| `task_1` | 0.82 | Inherited — no human reference exists |
| `task_2` | 0.82 | Confirmed — rejects 0.0% of human pairs |
| `task_3` | 0.73 | Revised — passes 95% of human pairs |

Structural-integrity checking (`T-107`) found zero blocking failures across all three
tasks — every block joins four-way with no label drift — while non-blocking findings
(script leakage, entity-placeholder leakage, nukta-induced escape leakage, suspected
truncation) were concentrated an order of magnitude more heavily in `task_1` than in
`task_2` or `task_3`, consistent with `task_1` being the most numeral- and entity-dense
of the three. Direction-reliability ranking (`T-111`) found that crossing *into* a
Dravidian language consistently costs more than crossing back into an Indo-Aryan one,
by 2.6–4.8 points, across all three tasks — the one finding in this ranking that
generalized across tasks, as opposed to two other candidate findings (whether Hindi→
Malayalam is worse than Bengali→Malayalam, and whether Dravidian-source directions
are worse generally) that reversed between tasks and were explicitly not reported as
general conclusions.

## B. Baseline Validation (Phase 2)

The Hindi in-language baseline was validated against IndicFinNLP's own published
numbers within a ±0.02 tolerance (Table III). `task_3`'s baseline passed comfortably
above its published figure, consistent with the paper's own low-data regime; `task_2`
passed below tolerance but was diagnosed rather than treated as a failure — after
hyperparameter tuning, the mean score was essentially unchanged but run-to-run
standard deviation tightened roughly tenfold, indicating the shortfall is a stable
property of this reproduction rather than noise, plausibly attributable to a smaller,
differently sampled test fold than the original paper's. The gate accordingly passed,
clearing Phase 3 to proceed.

**TABLE III. BASELINE VALIDATION GATE (HINDI, INDICBERT-V2)**

| Configuration | This project | Published | Result |
|---|---|---|---|
| `task2_hin_indicbert` | 0.823 ± 0.004 | 0.86 | Pass (diagnosed) |
| `task3_hin_indicbert` | 0.153 ± 0.025 | 0.05 | Pass (above) |

## C. Zero-Shot Transfer Loss (Phase 3, Hindi Source)

Zero-shot transfer out of Hindi loses roughly 0.39–0.45 macro-F1 on `task_2` and
roughly 0.50–0.51 on `task_3`, consistently across every target language and
comfortably larger than seed-to-seed variation (Table IV). Comparing native-target
against machine-translated-target evaluation conditions — which isolates whether the
loss is a genuine transfer failure or an artifact of this project's own translation
pipeline — the two land within a few points of each other on both tasks, evidence
that the drop reflects a real cross-lingual limitation of the model rather than
translation damage. A separate "translationese" comparison, which holds the target
language fixed at Hindi and varies only whether the text was natively written or
machine-translated into Hindi, shows a much smaller effect on `task_2` (−6 to −7
points) than on `task_3` (−26 to −27 points), which is attributed to `task_3`'s small
size and consequent sensitivity to any distribution shift, including one from
translation.

**TABLE IV. TRANSFER GAP, HINDI SOURCE, NATIVE TARGETS**

| Target | Task 2 gap | Task 3 gap |
|---|---|---|
| Bengali | 0.448 ± 0.072 | 0.510 ± 0.168 |
| Telugu | 0.453 ± 0.038 | 0.514 ± 0.170 |
| Malayalam | 0.387 ± 0.057 | 0.504 ± 0.184 |

## D. Encoder Comparison (Phase 5)

mBERT-base outperformed IndicBERT-v2 on every one of twelve comparable transfer
cells, by 0.15–0.37 macro-F1, a margin larger than seed-to-seed variation on both
encoders (Table V). The advantage does not generalize uniformly: on `task_2` mBERT
both starts higher in-language and loses substantially less under transfer, while on
`task_3` the two encoders lose nearly identical amounts, so "the better encoder
transfers better" was explicitly not reported as a general property.

**TABLE V. IN-LANGUAGE AND CROSS-LINGUAL MACRO-F1 BY ENCODER**

| | IndicBERT-v2 | mBERT |
|---|---|---|
| Task 2, in-language | 0.823 ± 0.004 | 0.874 ± 0.017 |
| Task 2, cross-lingual (mean of 6 cells) | 0.519 | 0.787 |
| Task 3, in-language | 0.153 ± 0.025 | 0.335 ± 0.042 |
| Task 3, cross-lingual (mean of 6 cells) | 0.087 | 0.294 |

Whether this advantage reflects mBERT's greater raw capacity or its broader
multilingual pretraining (capacity dilution) could not be resolved: the one
controlled comparison that would isolate the two factors — mBERT against XLM-R,
which share an identically sized transformer body and differ mainly in
pretraining breadth — is exactly the comparison blocked by the 4 GB GPU limit
described in Section III-E. A tokenizer-fragmentation measurement found IndicBERT-v2
needs 14–24% fewer tokens than mBERT for the same sentences, confirming its
specialization is real, yet it still underperformed on every task measure; the
resulting hypothesis — that the Indic model's cross-lingual representational
alignment, not its raw capacity, is the bottleneck — is recorded as unresolved
rather than established, since the 11× gap in the two encoders' actual
transformer-body sizes (once IndicBERT-v2's layer-weight-sharing is accounted for)
predicts a similar outcome on its own.

## E. Source-Language Comparison (Phase 4)

With the full reachable grid complete (126 of 126 planned results per task, across
both encoders), three findings replicated across both IndicBERT-v2 and mBERT and two
did not.

**Replicated findings:**

1. *Hindi is the worst source language of the three*, on both encoders, despite
   having the most training data and originally being this project's default choice
   (Table VI).
2. *The cost is concentrated in the target language, not the source/target pairing.*
   Crossing into a Dravidian language costs more than crossing into an Indo-Aryan one
   regardless of source, on both encoders — the same shape Phase 1 found independently
   in its translation-quality measurements, using an unrelated method.
3. *Transfer is directionally asymmetric.* Hindi→Bengali loses more than
   Bengali→Hindi on identical content with equally capable models, on both encoders,
   by a margin larger than seed noise.

**TABLE VI. MEAN TRANSFER GAP BY SOURCE LANGUAGE (TASK 2)**

| Source | IndicBERT-v2 | mBERT |
|---|---|---|
| Bengali | 0.345 | 0.153 |
| Telugu | 0.367 | 0.174 |
| Hindi | 0.429 | 0.196 |

**Non-replicated findings**, reported as encoder-specific rather than as properties
of the languages: whether same-family (typologically proximate) source languages
transfer better than cross-family ones agreed on only 1 of 3 comparable cases for
IndicBERT-v2 versus 3 of 3 for mBERT, with the two encoders directly contradicting
each other on Malayalam's best source language.

`task_3` supported no source comparison: every transfer gap on it clustered between
0.47 and 0.53 regardless of source, target, or encoder, consistent with the task
sitting at its data ceiling (532 rows across 10 classes; published baseline 0.05
macro-F1). A secondary finding from this phase — three of nine IndicBERT-v2 runs on
`task_3` never trained at all, and a fourth was stopped mid-improvement — is reported
rather than hidden, since a model that never learned scores equally badly at home and
abroad and would otherwise present as an implausibly good "transfer" result; mBERT
showed no equivalent failures under identical settings.

## F. Summary

Table VII condenses the headline quantitative results across phases.

**TABLE VII. HEADLINE RESULTS SUMMARY**

| Result | Value |
|---|---|
| Native-split independence (task 1 vs. tasks 2/3) | Independent (median 0.58–0.64) vs. parallel (0.88–0.91) |
| Calibrated τ (task 2 / task 3) | 0.82 / 0.73 |
| Hindi-source transfer gap (task 2 / task 3) | ~0.39–0.45 / ~0.50–0.51 |
| Best source language (both encoders) | Bengali |
| Worst source language (both encoders) | Hindi |
| Encoder comparison winner (12/12 cells) | mBERT over IndicBERT-v2 |
| Directional asymmetry (Indo-Aryan↔Dravidian) | 2.6–4.8 points, all three tasks |
| Evaluation coverage | 126 of 126 planned results per task |
| Diagnostic pipeline coverage | 1 of many conditions run end-to-end |

## G. Limitations of Results Obtained So Far

The results above are subject to limitations already identified and documented
during implementation, carried forward here rather than restated in full: no native
Malayalam text exists anywhere upstream, so every Malayalam result mixes transfer
difficulty with translation quality and has no human reference against which to
calibrate τ; no native speaker of Bengali, Telugu, or Malayalam has reviewed the
translated corpus beyond a 1–4%-of-rows spot check by a project maintainer; the LaBSE
similarity measure is a validated *relative* ranking tool but a poor *absolute*
quality score; neither the Bengali nor the Telugu baseline has been checked against a
published reference figure, unlike the Hindi baseline; and the three-encoder
comparison that would separate capacity from multilinguality as an explanation for
mBERT's advantage could not be run on the available hardware. Most importantly for
the immediate next steps, **no causal explanation for any transfer failure yet
exists** — Phases 1–5 establish the size and shape of the effect; explaining it is
the explicit purpose of the Phase 6 diagnostic pipeline described in Section III-G,
and of the not-yet-started Phase 7 aggregation of its output into findings.
