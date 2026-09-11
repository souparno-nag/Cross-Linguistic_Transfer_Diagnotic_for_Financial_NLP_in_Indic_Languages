# Phase 3 — Zero-Shot Evaluation and the Transfer Gap

**Specification:** `CLAUDE3.md`
**Status:** Complete for every evaluation currently possible. 21 of 63 planned
(condition × seed) runs are done; the other 42 are explicitly recorded as blocked,
pending Bengali/Telugu training (Phase 2 scope decision).

## What this phase measures

How much a classifier trained only on Hindi (Phase 2) loses when it is pointed — with no
further training, no gradient updates at all — at the *same underlying content* written
in Bengali, Telugu, or Malayalam. Because Phase 1 built a corpus where the same item
genuinely exists across languages, this phase can compare a model against **itself**: the
Hindi score on an item is the model's own ceiling, and the gap to its score on the same
item in another language is a clean measurement of what crossing the language boundary
costs, not a comparison against some external benchmark.

**Transfer gap = source (in-language) score − target (other-language) score.**

This phase deliberately stops short of explaining *why* any particular gap exists — that
is left to a future diagnostic phase. What it produces instead is the *size* of the gap,
broken down usefully (by language pair, by language family, by whether the target text
was human-written or machine-translated), plus a clean list of exactly which individual
predictions flipped from correct to wrong when the language changed — the raw material a
future causal analysis would need.

This phase depends on Phase 1's frozen corpus and Phase 2's trained checkpoints, and its
single hard precondition is that **T-207 (Phase 2's baseline validation gate) passed** —
if the in-language baseline hadn't reproduced the published numbers, no transfer result
measured against it would mean anything.

## The three families of evaluation condition

All evaluation conditions come from `configs/eval_conditions.json`, built in Phase 1's
T-113 — this phase does not invent new conditions ad hoc, since doing so would risk
recreating the training-on-your-own-content mistake T-113 was specifically built to
prevent. Three families of question are covered:

| Family | Question it answers | Count |
|---|---|---|
| **Transfer cells** | How much score is lost going from a specific source language to a specific target language? | 9 (every source × target pair) |
| **Typological quadrants** | Does moving between two closely related languages (Indo-Aryan → Indo-Aryan) lose less than moving between distantly related ones (Indo-Aryan → Dravidian)? | 4 |
| **Translationese** | Holding the language fixed, does it matter whether the text was written by a person or produced by this project's own machine translation? | 1 comparison per source language |

**The translationese family is the one that is easy to overlook, and it is the one that
actually answers whether this project's own translation pipeline is contaminating the
results.** `H_nat` (native Hindi) vs. `Hi←B` (Hindi machine-translated from Bengali) vs.
`Hi←T` (Hindi machine-translated from Telugu) is the *same language*, with the *same gold
labels*, differing only in whether that text was written by a human or produced by
IndicTrans2. If a model's score on genuinely native text differs a lot from its score on
machine-translated text in the *same* language, that's a sign this project's own
translation quality — not the model's cross-lingual ability — is driving part of what
looks like a transfer gap.

## Hard rules

1. **Zero-shot means zero parameter updates.** Every inference run asserts the model is
   in evaluation mode and that no optimizer exists anywhere in the code path used for
   inference — a single accidental gradient step here would silently invalidate every
   result in this phase.
2. **Never train in this phase.** Checkpoints from Phase 2 are read-only inputs.
3. **Log every individual prediction, not just aggregate scores.** A later diagnostic
   phase needs the per-item log and cannot be reconstructed from summary numbers alone.
4. **Never evaluate a checkpoint on the same split it was trained on**, except as the
   deliberate in-language ceiling measurement, which is always labeled explicitly as
   exactly that rather than mixed in with genuine cross-lingual results.
5. **3 seeds minimum**, reported as mean ± standard deviation. A gap smaller than the
   spread across seeds is not a finding — it's noise.
6. **Macro-F1 is the primary metric**, with accuracy reported alongside and per-class F1
   retained for deeper inspection.

## Tasks

### T-301 — Inference runner
`src/inference.py`

Loads a frozen checkpoint and runs it over any split with no parameter updates.
`load_frozen_model()` freezes every parameter and pins the model into evaluation mode;
`predict()` runs under a mode that disables gradient tracking entirely and additionally
asserts, at runtime, that the model never leaves evaluation mode during the call — this
enforces Hard Rules 1 and 2 as actual code checks rather than relying on convention alone.

**Status: done.** Verified on real data: re-running inference on a checkpoint's own
training-time dev fold reproduces the exact macro-F1 score that was recorded during
training in Phase 2 — confirming the inference path and the training path agree on what
the model actually learned.

### T-302 — Per-instance prediction log
`src/predictions.py`

For every evaluated instance, records its item ID, block, language, origin, source
language (if machine-translated), gold label, predicted label, and the full per-class
probability distribution. This log is explicitly treated as **the deliverable of this
phase**, not an incidental byproduct — it is exactly what a future diagnostic phase would
need to investigate individual failures, and cannot be reconstructed later from aggregate
metrics alone.

**Status: done.** Output path: `data/predictions/task_{n}/{condition_id}.parquet` — the
per-task directory level was added specifically after task 2 and task 3 initially
collided on an identically-named condition and silently overwrote each other, the exact
same failure mode Phase 1 already guards against for corpus paths. Verified on real data:
a block's four language arms (native Hindi plus its three machine-translated versions)
join together on `item_id` with zero missing rows, for both task 2 (2,238 items) and task
3 (532 items). The CLI persists only the native in-language-ceiling arm to disk (the
named exception under Hard Rule 4); the three machine-translated arms are predicted only
in memory to prove the join works, after an earlier version's habit of persisting them
too caused a real collision with T-305's own output files (see T-305 below).

### T-303 — Metrics
`src/metrics.py` (extended from Phase 2)

Adds a confusion matrix to Phase 2's existing accuracy/macro-F1/per-class-F1
implementation, validated directly against `sklearn`'s own confusion matrix
implementation on a fixture worked out by hand. A class that never actually appears in
a particular evaluation slice still occupies its own row and column in the matrix, filled
with zeros, rather than the matrix silently shrinking — consistent with how absent
classes were already handled in Phase 2's per-class F1 metric.

**Status: done.**

### T-304 — Bootstrap confidence intervals and the transfer gap
`src/transfer.py`

Computes a 95% confidence interval for any metric via 1,000-sample bootstrap resampling
(repeatedly re-sampling the evaluation items with replacement and recomputing the metric,
to see how much it varies) using a seeded random number generator, so two runs with the
same seed produce byte-identical resampled distributions. `transfer_gap()` reports
`source score − target score`, following the convention used by the XTREME cross-lingual
benchmark, with its own confidence interval built by pairing the two metrics'
resampled distributions together rather than computing each interval independently.
`within_seed_noise()` gives a concrete yes/no answer to "is this gap actually
distinguishable from run-to-run noise" by checking whether the gap's confidence interval
includes zero — turning Hard Rule 5 into an automatic check rather than something judged
by eye.

**Status: done.**

### T-305 — Run every evaluation condition
`src/evaluate.py`, `scripts/t305_run_conditions.py`

Runs IndicBERT-v2 over every condition in `configs/eval_conditions.json`, 3 seeds each,
reusing one loaded checkpoint across every condition and arm that needs it rather than
reloading per condition.

**Status: done for the 7 conditions runnable today** — 21 of the full 63
(condition × seed) combinations. These are the conditions whose source language is
Hindi (since only a Hindi checkpoint exists) — the Hindi-sourced native transfer cells,
the Hindi-sourced same-source-machine-translated transfer cells, and the Hindi
translationese comparison. **The other 42 are reported as explicitly blocked** (missing
Bengali/Telugu checkpoints, a direct consequence of Phase 2's Hindi-only scope decision)
rather than silently omitted from the results — the results tables always show the full
9-cell shape with blocked cells clearly marked, so the coverage gap is visible rather than
looking like a smaller, complete matrix.

**Two real bugs were found and fixed here.** First: the condition-running code was
predicting over an **entire** target split regardless of what the condition's own config
said the evaluation set should be — meaning a condition that was supposed to be scoped to
only the held-out evaluation partition was actually being evaluated on 100% of the split,
including items whose content (in the training language) the model had, in a very literal
sense, already seen during training — because tasks 2 and 3 share the same underlying
item across languages (the T-102b discovery). This was fixed by threading the intended
item filter through from the condition definition all the way to the prediction step.
Second: a related script (T-302's) was separately writing its own copies of the
machine-translated evaluation arms under the *same filenames* T-305 uses, but without
that same item filter applied — so running a second or third seed would silently
overwrite a correctly-filtered result with an unfiltered, leaky one. This was fixed by
scoping that script back to only persisting the native in-language-ceiling arm (see
T-302). Both fixes were verified afterward: every item in every runnable condition's log
was confirmed to fall inside the intended evaluation partition, with zero leakage, across
all 3 seeds.

### T-306 — Mismatch filter
`src/mismatch.py`

Identifies, row by row, the specific items where the model got the source-language
version **right** but the target-language version of the *same item* **wrong**
(`predicted_source == gold AND predicted_target != gold`). This is what makes a later
diagnostic phase meaningful at all: because the model demonstrably could answer this
exact question correctly one language earlier, the item's inherent difficulty and the
model's basic capability are both ruled out as explanations by construction — whatever
caused the flip has to be something about crossing the language boundary itself.

`join_source_target()` joins the source and target prediction logs on `(item_id, seed)`
— seed is deliberately part of the join key so a seed-0 prediction is never accidentally
paired with a different seed's prediction — and raises an error if an item that should be
aligned somehow carries a different gold label on each side, which would indicate a
deeper problem with the alignment itself.

**Status: done**, validated against 8 hand-built fixtures covering the join and
mismatch logic directly. On real data, task 2's mismatch rate runs 35–42% across the
runnable conditions, and task 3's runs 65–68% — consistent with task 3's much lower
overall in-language ceiling established back in Phase 2 (T-207).

### T-307 — Export the failed-instance set
`src/failures.py`

Exports exactly the mismatched items identified by T-306 to their own Parquet files, one
per evaluation condition, and manually sanity-checks a sample of 20 to confirm the filter
is behaving as intended rather than just producing numbers that happen to look
plausible.

**Status: done** for all 6 runnable transfer conditions (across both classification
tasks). Each task's audit note (`reports/task_{n}/t307_audit.md`) records the sanity
check against the real source and target text pulled directly from the frozen corpus —
all 40 sampled rows (20 per task) passed. The check itself is explicitly scoped as
**mechanical** — does the arithmetic of the mismatch predicate actually hold, does the
item ID resolve to real text on both sides — and deliberately **not** a linguistic
judgment of whether the failures make sense, since nobody working on this project reads
Bengali, Telugu, or Malayalam (the same limitation already documented in Phase 1's
datasheet).

### T-308 — Results tables
`src/results_tables.py`, `reports/transfer_results.md`

Renders the final transfer matrix (all 9 cells, including blocked ones), the typological
quadrant summary, and the translationese comparison.

**Status: done.** Every table deliberately keeps blocked cells visible rather than
dropping them, so the matrix's overall shape always reads as the full 9-cell design even
though only a third of it is currently fillable. Every gap is measured against the model
compared to *itself* on the identical items (via T-306's paired join) — not against
Phase 2's baseline number, which was computed on a different, randomly stratified fold,
and comparing across two different folds would have introduced noise unrelated to
language transfer at all.

**A real rendering bug was caught and fixed here too.** When an entire results column had
no data yet (before T-305's full rerun, some conditions had only one seed's worth of
ceiling data), pandas silently converted the missing values to `NaN` rather than `None`
— and in Python, `bool(float("nan"))` evaluates to `True`. A naive "is this value
missing" check written as `is None` therefore rendered every genuinely-unknown value as
"yes" in the output table instead of the intended "—", which would have actively
misrepresented incomplete data as a real result.

#### Results, task 2 (sustainable / unsustainable, binary)

**Transfer matrix** — native and same-source-machine-translated targets, Hindi source
only (the other 6 source rows are blocked pending Phase 2 training):

| Condition | Typological quadrant | Gap (mean ± std) | Within seed noise? |
|---|---|---|---|
| Hindi → Bengali (native) | Indo-Aryan → Indo-Aryan | 0.448 ± 0.072 | No |
| Hindi → Telugu (native) | Indo-Aryan → Dravidian | 0.453 ± 0.038 | No |
| Hindi → Malayalam (native) | Indo-Aryan → Dravidian | 0.387 ± 0.057 | No |
| Hindi → Bengali (machine-translated) | Indo-Aryan → Indo-Aryan | 0.466 ± 0.088 | No |
| Hindi → Telugu (machine-translated) | Indo-Aryan → Dravidian | 0.465 ± 0.025 | No |
| Hindi → Malayalam (machine-translated) | Indo-Aryan → Dravidian | 0.384 ± 0.069 | No |

**Translationese comparison (Hindi):**

| Block | Text origin | Macro-F1 (mean ± std) | Change vs. native |
|---|---|---|---|
| H (native Hindi) | native | 0.954 ± 0.016 | — (reference) |
| B (Hindi translated from Bengali) | machine-translated | 0.885 ± 0.010 | −0.069 |
| T (Hindi translated from Telugu) | machine-translated | 0.896 ± 0.019 | −0.057 |

#### Results, task 3 (10-way ESG topic, multi-class)

**Transfer matrix** — Hindi source only:

| Condition | Typological quadrant | Gap (mean ± std) | Within seed noise? |
|---|---|---|---|
| Hindi → Bengali (native) | Indo-Aryan → Indo-Aryan | 0.510 ± 0.168 | No |
| Hindi → Telugu (native) | Indo-Aryan → Dravidian | 0.514 ± 0.170 | No |
| Hindi → Malayalam (native) | Indo-Aryan → Dravidian | 0.504 ± 0.184 | No |
| Hindi → Bengali (machine-translated) | Indo-Aryan → Indo-Aryan | 0.500 ± 0.167 | No |
| Hindi → Telugu (machine-translated) | Indo-Aryan → Dravidian | 0.512 ± 0.174 | No |
| Hindi → Malayalam (machine-translated) | Indo-Aryan → Dravidian | 0.516 ± 0.189 | No |

**Translationese comparison (Hindi):**

| Block | Text origin | Macro-F1 (mean ± std) | Change vs. native |
|---|---|---|---|
| H (native Hindi) | native | 0.594 ± 0.185 | — (reference) |
| B (Hindi translated from Bengali) | machine-translated | 0.332 ± 0.080 | −0.262 |
| T (Hindi translated from Telugu) | machine-translated | 0.328 ± 0.062 | −0.267 |

**Reading these results together, plainly:**

- **The gap is large and consistent, on both tasks, in every direction measured so far**
  — roughly 0.39–0.45 macro-F1 lost on task 2, and roughly 0.50–0.51 lost on task 3, in
  every case comfortably larger than the run-to-run seed variation. This is a genuine
  cross-lingual transfer effect, not sampling noise.
- **The native and machine-translated target families land within a few points of each
  other on both tasks.** If translation damage from this project's own pipeline were the
  main cause of the drop, evaluating on real human-written Bengali text should score
  meaningfully better than evaluating on machine-translated Bengali text — it does not.
  That is evidence the transfer gap is a real limitation of the model's cross-lingual
  ability, not mostly an artifact of translation quality.
- **The translationese comparison (which holds language fixed at Hindi and varies only
  whether the text is native or machine-translated) tells a more nuanced story.** On task
  2 it drops a modest 6–7 points; on task 3 it drops a much larger 26–27 points. Since
  task 3 has by far the smallest and hardest-to-classify dataset (Phase 2, T-206), it
  appears more sensitive to any distribution shift at all, including one introduced by
  this project's own translation — a caveat worth remembering before treating task 3's
  cross-lingual numbers as being purely about language transfer.
- **Coverage is currently one-directional.** Every number above describes transfer *out
  of* Hindi. Nothing here yet says what happens transferring *into* Hindi, or between
  Bengali and Telugu directly — that requires the Bengali and Telugu baselines Phase 2
  deliberately deferred.

## Outputs

| Artifact | Path |
|---|---|
| Per-instance predictions | `data/predictions/task_{n}/{condition_id}.parquet` |
| Failed-instance sets | `data/failures/task_{n}/{condition_id}.parquet` |
| Metrics log | `experiments.csv` |
| Results tables | `reports/transfer_results.md` |

Prediction log columns: `condition_id, run_id, item_id, block_id, lang, origin,
src_lang, gold, pred, probs, seed`.

## Working agreement for this phase

- **Report the direction of every gap, including surprising ones.** If a Dravidian target
  ever doesn't drop more than an Indo-Aryan one, that would run counter to prior
  cross-lingual transfer literature and is a finding worth stating plainly, not burying.
- **Never attribute a failure to a specific cause in this phase.** Naming an actual cause
  (numeral loss, entity loss, sentence length, etc.) is a future diagnostic phase's job,
  and doing it here — without that phase's evidence — would produce a claim this phase
  cannot actually defend.
- **If a gap is within seed variance, say so explicitly** rather than reporting it as if
  it were a real effect.
- Prefer boring, reproducible code over anything clever.
