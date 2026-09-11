# CLAUDE.md — Phase 3: Zero-Shot Evaluation & Transfer Gap

Scope: **Phase 3 only.** Run frozen checkpoints over target splits, compute transfer
metrics, and emit the failed-instance set. Do **not** implement the diagnostic pipeline
(Phase 6) or attribute any failure to a linguistic cause. If a request drifts there, say
so and stop.

---

## Depends on

- `data/v1.0/` — frozen corpus, manifest-verified. Read-only.
- `checkpoints/{run_id}/` — frozen native-source checkpoints from Phase 2.
- **T-207 must have passed.** If in-language baselines did not reproduce IndicFinNLP's
  published numbers, stop. No transfer claim is valid on an unvalidated baseline.

Reuse `src/corpus_io.py`, `src/ids.py`, `configs/labels.json`, the Phase 2 loader.

---

## Goal

Measure how much performance is lost when a model trained on one Indic language is
applied to another, with content held constant by the parallel corpus.

**Transfer gap = source in-language score − target score.** The source split is both the
training data and the in-language ceiling, so the gap is measured against the model's own
capability, not an external reference.

---

## Evaluation conditions

Read `configs/eval_conditions.json` (Phase 1, T-113). Do not construct conditions
ad hoc. Three families:

| Family | Question it answers |
| --- | --- |
| **Transfer cells** (9) | How much is lost going source → target? |
| **Typological quadrants** (4) | Does Indo-Aryan↔Indo-Aryan beat Indo-Aryan↔Dravidian? |
| **Translationese** | Same language, native vs MT provenance — how much of the gap is caused by text being translated at all? |

The translationese family is the one people forget. `H_nat` vs `Hi←B` vs `Hi←T` is the
same language with the same labels, differing only in origin. It isolates the
translation penalty from the language penalty and is what separates a real transfer
finding from an artefact of your own MT.

---

## Hard rules

1. **Zero-shot means zero updates.** Assert `model.training is False` and that no
   optimiser exists in the inference path. A gradient step here silently invalidates
   everything.
2. **Never train in this phase.** Checkpoints are read-only inputs.
3. **Log every instance**, not just aggregates. Phase 6 consumes the per-instance log
   and cannot be rebuilt from summary metrics.
4. **Never evaluate a checkpoint on its own training split** except as the deliberate
   in-language ceiling, and label that condition explicitly.
5. **3 seeds**, mean ± std. A gap smaller than seed variance is not a finding.
6. **Macro-F1 primary**, accuracy alongside, per-class F1 retained.

---

## Tasks

### T-301 — Inference runner

Load a frozen checkpoint, run over any split, no parameter updates.
**Done:** asserts no gradient step occurs; re-running on the training split reproduces
the Phase 2 dev number.

**Status: DONE.** `src/inference.py`. `load_frozen_model` freezes every parameter and
pins the model to eval mode; `predict` asserts, under `torch.no_grad()`, that the model
never leaves eval mode and builds no gradient graph — hard rules 1-2 enforced, not just
followed by convention. Acceptance verified on real data: re-evaluating a trained
checkpoint on its own dev fold reproduces the exact macro-F1 recorded at training time.

### T-302 — Per-instance prediction log

Write `item_id`, `block_id`, `lang`, `origin`, `src_lang`, gold, predicted, per-class
probabilities for every instance.
**Done:** the log joins 4-way on `item_id` within a block with zero nulls.
This artefact is the input to Phase 6 — treat it as a deliverable, not a byproduct.

**Status: DONE.** `src/predictions.py`, `scripts/t302_predictions.py`,
`data/predictions/task_{n}/{condition_id}.parquet` — the `task_{n}` level was added after
task 2 and task 3 briefly collided on an identically-named condition and silently merged
(same failure mode CLAUDE.md §5 warns about for corpus paths). Verified on real corpus
data: block H's four arms (native Hindi plus its three MT translations) join on `item_id`
with zero nulls, for both task 2 (2238 items) and task 3 (532 items). The CLI persists
only the native in-language-ceiling arm (hard rule 4's named exception); the three MT
arms are predicted in memory purely to prove the join, after an earlier version's habit
of also persisting them collided with T-305's condition files (see T-305).

### T-303 — Metrics

Accuracy, macro-F1, per-class F1, confusion matrix.
**Done:** validated against `sklearn` on a fixture.

**Status: DONE.** Extended Phase 2's `src/metrics.py` (accuracy/macro-F1/per-class F1
already existed) with a fourth key, `confusion_matrix`, validated directly against
`sklearn.metrics.confusion_matrix` on a fixture worked out by hand, plus the same
absent-class treatment `per_class_f1` already used (an unseen class still occupies a
row/column, as zero, rather than shrinking the matrix).

### T-304 — Bootstrap CIs and transfer gap

1000-sample bootstrap for 95% CI; gap = source − target, XTREME reporting convention.
**Done:** reproducible under a fixed seed.

**Status: DONE.** `src/transfer.py`. `bootstrap_metric` resamples item indices with
replacement via a seeded `numpy.random.Generator` — two calls with the same seed return
byte-identical 1000-sample arrays. `transfer_gap` reports `source − target` (XTREME's
convention) with its own CI, built by pairing the two resampled-metric distributions;
`within_seed_noise()` flags a gap whose CI straddles zero, operationalising hard rule 5
directly rather than leaving "is this a finding" to eyeballing.

### T-305 — Run all evaluation conditions

IndicBERT-v2 × every condition in `eval_conditions.json` × 3 seeds.
**Done:** every condition has a result row in `experiments.csv`.

**Status: DONE for the 7 conditions runnable today** (21 of 63 (condition, seed) pairs —
the Hindi-sourced native and same-source-MT transfer cells, plus `translationese_hin`).
The other 42 are reported as explicitly blocked (missing Bengali/Telugu checkpoints,
deferred per CLAUDE2.md) rather than silently skipped. `src/evaluate.py` reads the
committed matrix, reuses one loaded checkpoint across every condition and arm that needs
it, and writes both `experiments.csv` and T-302's prediction log per arm.

**Two real bugs found and fixed here, both now verified clean.** First: `run_condition`
predicted over an entire target split regardless of `eval_conditions.json`'s
`eval_items`, so a `partition:eval`-scoped condition was actually evaluated on 100% of
the split — including items whose content, in the training language, the model had
literally trained on (tasks 2/3 share `item_id` across languages for an aligned item,
T-102b). Fixed by threading an item filter from the condition's own `eval_items` through
to prediction. Second: `scripts/t302_predictions.py` separately persisted its MT arms
under these same condition filenames without that filter, so running it for a second or
third seed silently overwrote the correctly-filtered seed's siblings with unfiltered
ones. Fixed by scoping that script back to persisting only the native-ceiling arm.
Verified after both fixes: every item in every runnable condition's log falls in the
`eval` partition, zero leaks, across all 3 seeds.

### T-306 — Mismatch filter predicate

Row-wise over the joined log: `ŷ_src == y AND ŷ_tgt != y`.
**Done:** unit-tested against hand-built fixtures; per-condition counts reported.
This is what makes Phase 6 meaningful — the model answered this exact sentence correctly
one language earlier, so incapacity and item difficulty are ruled out by construction.

**Status: DONE.** `src/mismatch.py`, 8 hand-built fixture tests. `join_source_target`
joins on `(item_id, seed)` — seed is part of the key so a seed-0 source prediction is
never paired with a different seed's target — and raises if an aligned item somehow
carries conflicting gold labels across the two logs. `scripts/t306_mismatches.py`
reports real per-condition, per-seed counts: task 2's mismatch rate runs 35-42% across
the native and same-source-MT families; task 3's runs 65-68%, consistent with its much
lower overall in-language ceiling (T-207).

### T-307 — Emit failed-instance set

Export failures as Parquet; manually sanity-check 20 to confirm the filter behaves as
intended.
**Done:** `data/failures/{condition}.parquet`; audit note written.

**Status: DONE.** `src/failures.py`, `scripts/t307_failures.py`,
`data/failures/task_{n}/{condition}.parquet` for all 6 runnable transfer/transfer_mt
conditions per task (namespaced by task for the same reason T-302's logs are). Each
task's `reports/task_{n}/t307_audit.md` records a 20-row sanity check with the real
source/target text pulled from the frozen corpus — all 40 sampled rows (20 per task)
pass. The check is explicitly scoped as mechanical (does the predicate's arithmetic
hold, does the item_id resolve to real text on both sides), not linguistic — nobody on
this project reads Bengali, Telugu or Malayalam (CLAUDE.md's second T-114 limitation).

### T-308 — Results tables

Transfer matrix (all 9 cells), quadrant summary, translationese comparison.
**Done:** tables render for the Review 3 deck.

**Status: DONE.** `src/results_tables.py`, `scripts/t308_results.py` →
`reports/transfer_results.md`. Every table keeps blocked cells rather than dropping
them, so the matrix's shape is always the full 9. The gap compares the model against
itself on identical items (T-306's paired join), not against T-207's baseline number
computed on a different, randomly stratified fold. Every runnable cell now has real
3-seed mean ± std and comes back **"not within seed noise" in every case measured so
far** — a genuine finding, not sampling noise:

| | task 2 (binary) | task 3 (10-class) |
|---|---|---|
| native family gap | 0.39 – 0.45 | 0.50 – 0.51 |
| same-source-MT family gap | 0.38 – 0.47 | 0.50 – 0.52 |

The two families landing so close together within each task is itself a small finding:
whether the target text is real Bengali/Telugu/Malayalam or machine-translated from the
same Hindi source barely moves the size of the gap. A rendering bug was also caught and
fixed here: when a column's values were all `None` (every condition had only 1 seed of
ceiling data before T-305's rerun), pandas silently coerced it to NaN, and
`bool(float("nan"))` is `True` in Python — a naive `is None` check rendered every
genuinely-unknown value as "yes" instead of "—".

---

## Outputs

| Artefact | Path |
| --- | --- |
| Per-instance predictions | `data/predictions/{condition}.parquet` |
| Failed instances | `data/failures/{condition}.parquet` |
| Metrics | `experiments.csv` |
| Results tables | `reports/transfer_results.md` |

Prediction log columns: `condition_id, run_id, item_id, block_id, lang, origin,
src_lang, gold, pred, probs, seed`.

---

## Working agreement

- **Report the direction of every gap, including surprising ones.** If Dravidian targets
  don't drop more than Indo-Aryan ones, that contradicts Pires et al. and is a finding.
  Do not bury it.
- **Never attribute a failure to a cause here.** Naming causes is Phase 6, and doing it
  early without the module evidence produces claims you cannot defend.
- **If a gap is within seed variance, say so** rather than reporting it as an effect.
- Prefer boring code. This must be reproducible cold in October.
