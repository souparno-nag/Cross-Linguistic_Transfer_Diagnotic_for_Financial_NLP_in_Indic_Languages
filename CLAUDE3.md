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

### T-302 — Per-instance prediction log

Write `item_id`, `block_id`, `lang`, `origin`, `src_lang`, gold, predicted, per-class
probabilities for every instance.
**Done:** the log joins 4-way on `item_id` within a block with zero nulls.
This artefact is the input to Phase 6 — treat it as a deliverable, not a byproduct.

### T-303 — Metrics

Accuracy, macro-F1, per-class F1, confusion matrix.
**Done:** validated against `sklearn` on a fixture.

### T-304 — Bootstrap CIs and transfer gap

1000-sample bootstrap for 95% CI; gap = source − target, XTREME reporting convention.
**Done:** reproducible under a fixed seed.

### T-305 — Run all evaluation conditions

IndicBERT-v2 × every condition in `eval_conditions.json` × 3 seeds.
**Done:** every condition has a result row in `experiments.csv`.

### T-306 — Mismatch filter predicate

Row-wise over the joined log: `ŷ_src == y AND ŷ_tgt != y`.
**Done:** unit-tested against hand-built fixtures; per-condition counts reported.
This is what makes Phase 6 meaningful — the model answered this exact sentence correctly
one language earlier, so incapacity and item difficulty are ruled out by construction.

### T-307 — Emit failed-instance set

Export failures as Parquet; manually sanity-check 20 to confirm the filter behaves as
intended.
**Done:** `data/failures/{condition}.parquet`; audit note written.

### T-308 — Results tables

Transfer matrix (all 9 cells), quadrant summary, translationese comparison.
**Done:** tables render for the Review 3 deck.

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
