# CLAUDE.md — Phase 2: Training Pipeline & In-Language Baselines

Scope: **Phase 2 only.** Build the fine-tuning pipeline and establish in-language
baselines. Do **not** implement cross-lingual evaluation (Phase 3) or the diagnostic
pipeline (Phase 6). If a request drifts there, say so and stop.

---

## Depends on

`data/v1.0/` — frozen, hash-verified corpus from Phase 1. Read-only. Verify the
manifest at startup and refuse to run if it fails.

Reuse `src/corpus_io.py`, `src/ids.py`, `configs/labels.json`. Do not reimplement them.

---

## Goal

Fine-tune an encoder on a **native** split and reproduce IndicFinNLP's published
monolingual baseline. Nothing about cross-lingual transfer is claimed or measured until
this reproduces.

Three baselines: `H_nat` (Hindi), `B_nat` (Bengali), `T_nat` (Telugu).
Malayalam has no native split and gets no baseline.

---

## Encoders

| Model | HF id | Status |
| --- | --- | --- |
| IndicBERT-v2 | `ai4bharat/indic-bert` | committed |
| XLM-R base | `xlm-roberta-base` | Phase 5 extension |
| mBERT base | `bert-base-multilingual-cased` | Phase 5 extension |

Phase 2 only needs IndicBERT-v2 working. Keep the loader model-agnostic so Phase 5 costs
nothing.

---

## Hard rules

1. **Train on native splits only.** MT splits are evaluation data. Training on them
   contaminates Phase 3.
2. **Never write to `data/v1.0/`.**
3. **3 seeds minimum** per configuration. Report mean ± std, never a single number.
4. **Log the config hash** with every run.
5. **Deterministic.** Same config + same seed → same metrics.
6. **Macro-F1 is primary.** Accuracy is reported alongside but never alone — class
   imbalance hides failure.

---

## Model

Pooled `h[CLS]` → linear head over K classes → cross-entropy. That is the whole
architecture. Do not add layers, pooling strategies, or loss tricks.

---

## Tasks

### T-201 — Environment

Python 3.11, pinned `requirements.txt`, CUDA verified on both machines.
**Done:** identical `pip freeze` on the 3060 and 3050; smoke test passes on both.

### T-202 — Dataset loader

Loads any split by `(block_id, lang, origin)`; tokenizes per encoder; asserts labels
against `labels.json`.
**Done:** unit tests cover all 4 languages × 3 encoders.

### T-203 — Training loop

Linear head, cross-entropy, AdamW, warmup, early stopping on dev macro-F1.
**Done:** overfits a 50-example subset to >0.95 train F1. This is a correctness check,
not a result.

### T-204 — Config system

YAML: seed, LR, batch size, max_len, epochs, encoder, split. Config hash written into
every result row.
**Done:** two runs of one config produce identical metrics.

### T-205 — VRAM budgets

Max batch × max_len per encoder on 12 GB and 4 GB, with fp16 and gradient accumulation.
**Done:** documented budget table; no OOM across a full epoch on either machine.

### T-206 — Baseline runs

IndicBERT-v2 on `H_nat`, `B_nat`, `T_nat` × 3 seeds = 9 runs.
**Done:** mean ± std macro-F1 per language logged to `experiments.csv`; checkpoints
saved.

### T-207 — Baseline validation gate ⚠️

Compare each in-language macro-F1 to IndicFinNLP's published monolingual baseline.
**Done:** within ±2 F1, or a written diagnosis of the gap.
**Nothing in Phase 3 starts until this passes.** A gap here usually means tokenization
or label-mapping error, not a modelling problem.

### T-208 — Checkpoint versioning

Naming convention, storage path, retention policy.
**Done:** any run's checkpoint retrievable from its `run_id`.

---

## Outputs

| Artefact | Path |
| --- | --- |
| Run log | `experiments.csv` |
| Checkpoints | `checkpoints/{run_id}/` |
| Configs | `configs/train/*.yaml` |
| Baseline report | `reports/baselines.md` |

`experiments.csv` columns: `run_id, encoder, split, seed, config_hash, accuracy,
macro_f1, date`.

---

## Working agreement

- Report the baseline gap honestly. Do not tune until it matches — diagnose first.
- If a run OOMs, report the config. Do not silently reduce batch size.
- Prefer boring code. This must be reproducible cold in October.
