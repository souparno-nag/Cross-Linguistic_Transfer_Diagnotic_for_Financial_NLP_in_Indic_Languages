# CLAUDE.md — Phase 4: Bengali & Telugu Source Training + Zero-Shot Evaluation

Scope: **Phase 4 only.** Train on the Bengali and Telugu native splits and evaluate
zero-shot across their parallel targets, for all three encoders. Do **not** build or
modify diagnostic modules (Phase 6) or aggregate root-cause findings (Phase 7). If a
request drifts there, say so and stop.

---

## What this phase completes

Phase 2/3 covered **block H** (Hindi source). Phase 5 adds the remaining encoders for
block H. Phase 4 fills the rest of the grid:

| | IndicBERT-v2 | XLM-R | mBERT |
| --- | --- | --- | --- |
| **Block H** (Hindi source) | Phase 2/3 | Phase 5 | Phase 5 |
| **Block B** (Bengali source) | **Phase 4** | **Phase 4** | **Phase 4** |
| **Block T** (Telugu source) | **Phase 4** | **Phase 4** | **Phase 4** |

Phase 4 owns **6 of the 9 cells** and is the single largest compute block in the
project. Plan accordingly — see the budget warning below.

### Two knock-on effects to confirm before starting

- **Phase 2's scope narrows.** Its baseline task named `H_nat`, `B_nat` and `T_nat`.
  Under this split, Phase 2 delivers `H_nat` only; `B_nat` and `T_nat` move here.
  Confirm this is the intent before running anything.
- **Review 3 delivery has lost its slot.** It was the old Phase 4. It still has to
  happen on 16 September and now sits outside the numbered phases — decide where it
  lives.

---

## GPU budget warning

Six encoder×block cells × 3 seeds ≈ **25–30 GPU-hours** for training, plus inference.
Against roughly 10 usable GPU-hours a week, this phase alone is three weeks of lab
access, and Phases 5 and 6 are competing for the same card.

**Therefore: baseline-first, strictly.** Complete `B_nat` × IndicBERT-v2 end to end
before starting any other cell. A partially-filled grid with one complete row is a
defensible result; six half-finished cells is not.

Suggested cell order:

```md
1. B_nat × IndicBERT-v2     ← prove the cell works end to end
2. T_nat × IndicBERT-v2     ← completes the IndicBERT row across all blocks
3. B_nat × XLM-R
4. T_nat × XLM-R
5. B_nat × mBERT
6. T_nat × mBERT
```

Stop at any point and the result is still reportable.

---

## Depends on

- Phase 2 pipeline: loader, training loop, config system, VRAM budgets
- Phase 3 pipeline: inference runner, metrics, bootstrap CIs, mismatch filter
- `configs/eval_conditions.json` — Phase 1, T-113
- `data/v1.0/` — frozen corpus, read-only

**This phase should be configuration, not code.** Adding a new source block is a config
entry. If it isn't, report that rather than working around it.

---

## Transfer directions covered

| Source | Targets | Typology |
| --- | --- | --- |
| Bengali (`B_nat`) | Hindi, Malayalam, Telugu | IA→IA, IA→Dr, IA→Dr |
| Telugu (`T_nat`) | Hindi, Bengali, Malayalam | Dr→IA, Dr→IA, **Dr→Dr** |

`Te→Ml` is the Dravidian→Dravidian cell — the one condition with no precedent in your
literature review. Do not let it get deprioritised because it sits last in a loop.

Together with block H, this gives bidirectional coverage of Indo-Aryan↔Indo-Aryan and
Indo-Aryan↔Dravidian, which lets you test whether transfer is **symmetric** — a question
Pires et al. [12] does not settle.

---

## Hard rules

1. **Identical protocol to Phases 2/3 and 5.** Same seeds, same metric definitions, same
   config shape. Anything that moves besides source block and encoder invalidates the
   comparison.
2. **Train on native splits only.** `B_nat` and `T_nat`. MT splits are evaluation data —
   training on `Bn←H` contaminates every transfer number in block H.
3. **Zero-shot means zero updates.** Assert it, as in Phase 3.
4. **3 seeds**, mean ± std.
5. **Append-only** to `experiments.csv`; namespace by `encoder_id` and `block_id`.
6. **No hyperparameter tuning per block.** If you tune one, tune all and say so.

---

## Tasks

### T-401 — Source-block smoke test

Add `B_nat` as a training source via config and run one epoch on a 200-row subset,
end to end through training and inference.
**Done:** completes with no code change. If it doesn't, stop and report what is
hardcoded to block H.

### T-402 — Bengali baseline ⚠️ GATE

Fine-tune IndicBERT-v2 on `B_nat`, 3 seeds, evaluate in-language.
**Done:** macro-F1 within ±2 of IndicFinNLP's published Bengali monolingual baseline, or
a written diagnosis of the gap. **No Bengali transfer claim proceeds until this passes.**
GPU ≈4h.

### T-403 — Telugu baseline ⚠️ GATE

Same for `T_nat`.
**Done:** within ±2 of the published Telugu baseline, or a diagnosis. GPU ≈4h.

### T-404 — Bengali-source zero-shot evaluation

IndicBERT-v2 → Hindi, Malayalam, Telugu, plus the in-language ceiling and the Bengali
translationese conditions (`B_nat` vs `Bn←H` vs `Bn←T`).
**Done:** all conditions logged; per-instance predictions written.

### T-405 — Telugu-source zero-shot evaluation

Same for block T, including `Te→Ml`.
**Done:** as above.

### T-406 — Extend to XLM-R and mBERT

Repeat T-402 through T-405 for the remaining two encoders, in the cell order above.
**Done:** every completed cell has baseline + full evaluation logged. Partial completion
is acceptable and must be recorded as such — state which cells ran, not just which
succeeded. GPU ≈17h.

### T-407 — Emit failure sets for Phase 6

Run the Phase 3 mismatch filter over all new predictions.
**Done:** `data/failures/{condition}.parquet` with `encoder_id` and `block_id`
populated. **Coordinate with Phase 6 first** — they need T-601 finished.

### T-408 — Source-selection comparison table

All source blocks × all encoders × all targets, mean ± std, gaps with CIs.
**Done:** `reports/source_comparison.md`.

This table is what operationalises Lin et al. [16] on transfer-language selection. For
each target language, which source transfers best — and does typological proximity
predict it? That is a result your original single-source design could only cite, not
test.

---

## Outputs

| Artefact | Path |
| --- | --- |
| Run log | `experiments.csv` (append-only) |
| Checkpoints | `checkpoints/{run_id}/` |
| Predictions | `data/predictions/{condition}.parquet` |
| Failure sets | `data/failures/{condition}.parquet` |
| Comparison | `reports/source_comparison.md` |

`encoder_id` and `block_id` populated on every row of every artefact.

---

## Working agreement

- **Baseline-first is not advisory here.** The GPU budget does not cover the full grid
  comfortably. One complete row beats six partial cells.
- **Report which cells did not run.** An incomplete grid stated plainly is a scoping
  decision; an incomplete grid presented as complete is a defect.
- **Report asymmetry honestly.** If `Hi→Bn` and `Bn→Hi` differ substantially, that is a
  finding, not an error to be explained away.
- Prefer boring code. This must be reproducible cold in October.
