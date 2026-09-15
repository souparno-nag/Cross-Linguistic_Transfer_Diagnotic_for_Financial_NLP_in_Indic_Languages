# CLAUDE.md — Phase 5: Encoder Extension

Scope: **Phase 5 only.** Extend the existing training and evaluation pipeline to XLM-R
and mBERT. Do **not** modify the diagnostic pipeline (Phase 6, in progress in parallel)
or aggregate root-cause findings (Phase 7). If a request drifts there, say so and stop.

---

## Running in parallel with Phase 6 — read this first

Phase 6 is being built at the same time by a different member. Both phases touch shared
artefacts. Three rules keep them from colliding:

1. **Namespace everything by `encoder_id`.** Never write to a path or row that another
   encoder's run owns. Append to `experiments.csv`; never rewrite it.
2. **Do not edit anything under `src/diagnostics/`.** That is Phase 6's tree. If a
   diagnostic module needs a change to accommodate a new tokenizer, raise it — do not
   patch it yourself.
3. **GPU is the shared constraint, not code.** Phase 5 needs ~11 GPU-hours and Phase 6's
   Integrated Gradients module is also GPU-bound, against roughly 10 GPU-hours a week.
   Agree a schedule before starting; queue jobs so neither blocks interactively.

Phase 6 is the higher priority of the two. If GPU time is contested, Phase 6 wins —
it is on the critical path and Phase 5 is not.

---

## Depends on

- Phase 2 pipeline: loader, training loop, config system, VRAM budgets
- Phase 3 pipeline: inference runner, metrics, bootstrap CIs, `eval_conditions.json`
- `data/v1.0/` — frozen corpus, read-only

**This phase should be almost entirely configuration.** If adding an encoder requires
new code beyond a config entry, Phases 2 and 3 were not written model-agnostically —
report that immediately rather than working around it. T-500 exists to find out.

---

## Encoders

| Model | HF id | Role |
| --- | --- | --- |
| IndicBERT-v2 | `ai4bharat/indic-bert` | done in Phase 2/3 — the reference |
| XLM-R base | `xlm-roberta-base` | tests capacity dilution against an Indic-specialised model |
| mBERT base | `bert-base-multilingual-cased` | legacy baseline; the model Pires et al. [12] probed |

---

## Hard rules

1. **Identical protocol across encoders.** Same splits, same conditions, same seeds,
   same metric definitions. A comparison is only valid if nothing else moved.
2. **Do not tune hyperparameters per encoder** beyond what VRAM forces. If you tune one,
   you must tune all three and say so.
3. **Train on native splits only.** MT splits are evaluation data.
4. **3 seeds**, mean ± std.
5. **Append-only** to `experiments.csv`.
6. **Report OOM configs**; do not silently shrink batch size.

---

## Tasks

### T-500 — Model-agnosticism smoke test

Add XLM-R as a config entry and run one epoch on a 200-row subset, end to end through
training and inference.
**Done:** completes with no code change. If it does not, stop and report what is
hardcoded — fixing that is cheaper now than after 11 GPU-hours of runs.

### T-501 — XLM-R: train and evaluate

Native splits × 3 seeds, then every condition in `eval_conditions.json`.
**Done:** all result rows in `experiments.csv`; per-instance predictions written to
`data/predictions/`. GPU ≈6h.

### T-502 — mBERT: train and evaluate

Same protocol.
**Done:** as above. GPU ≈5h.

### T-503 — MAD-X adapter fallback (contingency)

**Only if T-501 or T-502 is VRAM-blocked.** Language and task adapters per Pfeiffer
et al. [17], which is compute-feasible on the 4 GB card.
**Done:** adapter run completes, or the task is closed as unnecessary with a note.

### T-504 — Cross-encoder comparison table

All three encoders × all conditions, mean ± std, gaps with confidence intervals.
**Done:** one table; `reports/encoder_comparison.md`.

### T-505 — Capacity-dilution analysis

Does the Indic-specialised encoder beat the 100-language model, and where? Ties to
Conneau et al. [13], who describe the positive-transfer versus capacity-dilution
trade-off.
**Done:** written analysis with supporting numbers. Report it faithfully if XLM-R wins —
that contradicts the motivation for choosing an Indic encoder and is worth stating.

### T-506 — Emit failure sets for Phase 6

Run the Phase 3 mismatch filter over the new predictions so Phase 6 can label them.
**Done:** `data/failures/{condition}.parquet` written with `encoder_id` populated.
**Coordinate with Phase 6 before running** — they must have finished T-601 first.

---

## Outputs

| Artefact | Path |
| --- | --- |
| Run log | `experiments.csv` (append-only) |
| Checkpoints | `checkpoints/{run_id}/` |
| Predictions | `data/predictions/{condition}.parquet` |
| Failure sets | `data/failures/{condition}.parquet` |
| Comparison | `reports/encoder_comparison.md` |

`encoder_id` must be populated on every row of every artefact.

---

## Working agreement

- **This phase is deferrable.** If time runs short, a single-encoder project is complete
  and defensible. Do not let it consume Phase 6's GPU budget.
- **Report a lost comparison honestly.** If mBERT cannot be fine-tuned within VRAM and
  adapters were used instead, that is a protocol difference and must be stated, not
  smoothed over.
- Prefer boring code. This must be reproducible cold in October.
