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

**Hindi only for now.** Because of time constraints, Phase 2 fine-tunes on `H_nat`
(Hindi) alone. `B_nat` (Bengali) and `T_nat` (Telugu) baselines are deferred, not
cancelled — keep the loader, config system and training loop language-agnostic so
adding them later costs only GPU time. Malayalam has no native split and gets no
baseline regardless.

---

## Encoders

| Model | HF id | encoder key | Status |
| --- | --- | --- | --- |
| IndicBERT-v2 | `ai4bharat/indic-bert` | `indicbert-v2` | committed |
| XLM-R base | `xlm-roberta-base` | `xlm-r-base` | Phase 5 extension |
| mBERT base | `bert-base-multilingual-cased` | `mbert-base` | Phase 5 extension |

Phase 2 only needs IndicBERT-v2 working. Keep the loader model-agnostic so Phase 5 costs
nothing — `src/data.py` holds the registry and both other encoders already load and
tokenise.

**`ai4bharat/indic-bert` is gated**, like the IndicTrans2 repos (CLAUDE.md §3.1). Being
logged in to the Hub is not enough: accept the terms at
<https://huggingface.co/ai4bharat/indic-bert> with the same account, then the download
works. Until then the loader raises `OSError: gated repo` and the `slow` tokeniser tests
for `indicbert-v2` skip with that reason rather than failing.

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

## Code layout

Same shape as Phase 1: **all logic lives in importable `src/` modules**
(`src/train.py`, `src/data.py`, `src/config.py`, `src/metrics.py`, …), each with
tests in `tests/`, driven by thin `scripts/t20x_*.py` CLIs run as
`python -m scripts.t20x_*`. The corpus modules stay flat in `src/`.

A **single** exploratory notebook, `notebooks/phase2.ipynb`, is allowed — but only
as a *driver*: it imports `src/` code to launch runs and inspect results (VRAM
probing, tokeniser output, the T-207 baseline gap, learning curves). No pipeline
logic is defined in it. Nothing on the reproducibility path — the loader, the
training loop, the config system, the baseline runs — may exist only in the
notebook, because out-of-order cell execution and hidden kernel state defeat
`hard rule 5` and the "reproducible cold in October" requirement. Strip outputs
before committing the notebook.

---

## Tasks

### T-201 — Environment

**Python 3.11 exactly** — assert it at every entry point and fail loudly on mismatch.
Phase 2 shares the Phase 1 venv (`env/`, gitignored), so `transformers` stays pinned
`<5` (CLAUDE.md §3.2). IndicBERT-v2 runs fine on 4.x; do not upgrade to satisfy a
Phase 2 dependency without re-reading that section. Run everything as a module from the
repo root (`python -m src.…`, `python -m scripts.…`). Pin every new dependency in
`requirements.txt` with a one-line reason, matching the existing entries.

`src/env_check.py` is the guard: every Phase 2 entry point calls
`require_python()` before doing any work, and `python -m src.env_check` is the
smoke test — it checks the interpreter, imports the training stack, and reports
versions and CUDA state, exiting non-zero on any problem.

**Done.** `src/env_check.py` (+ `tests/test_env_check.py`); `python -m src.env_check`
passes on the 3050; `requirements.txt` carries the Phase 2 deps and
`requirements.lock.txt` is the committed full `pip freeze`.

### T-202 — Dataset loader
`src/data.py`, `tests/test_data.py`

Loads any split by `(block_id, lang, origin)`; tokenizes per encoder; asserts labels
against `labels.json`.

`load_split(task, block, lang, origin)` goes through `src/corpus_io.read_split`, so the
§6 schema and label checks run before a row is seen. `origin` is asserted against the
file's actual contents — asking for `native` on a machine-translated split raises rather
than returning nothing, so `hard rule 1` cannot be broken by accident. The numeral task
(task 1) is refused: it has no classification label. `stratified_split()` makes the
seeded, class-stratified train/dev/test partition the frozen corpus does not carry.
`SplitDataset` tokenises once with truncation and pads per batch in `collate` to keep
wasted compute off the 4 GB card. The encoder registry (`ENCODERS`) carries all three
models; only IndicBERT-v2 matters for Phase 2.

**Done.** 22 offline tests (origin assertion, task-1 refusal, label range, deterministic
stratified split) plus `slow` tests tokenising all 4 languages × 3 encoders — xlm-r and
mbert pass; the 4 `indicbert-v2` cases skip until its gated repo is accepted (see
Encoders).

### T-203 — Training loop

Linear head, cross-entropy, AdamW, warmup, early stopping on dev macro-F1.
**Done:** overfits a 50-example subset to >0.95 train F1. This is a correctness check,
not a result.

### T-204 — Config system

YAML: seed, LR, batch size, max_len, epochs, encoder, split. Config hash written into
every result row.
**Done:** two runs of one config produce identical metrics.

### T-205 — VRAM budgets

Max batch × max_len per encoder, with fp16 and gradient accumulation. Set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before torch is imported — on the
4 GB card, without it training OOMs on fragmentation rather than genuine exhaustion
(CLAUDE.md §3.3).
**Done:** documented budget table; no OOM across a full epoch.

### T-206 — Baseline runs

IndicBERT-v2 on `H_nat` × 3 seeds = 3 runs. (`B_nat` and `T_nat` deferred — see Goal.)
GPU access is intermittent (CLAUDE.md §3): every run must be resumable and checkpoint
partial progress, so an interrupted run resumes instead of restarting.
**Done:** mean ± std macro-F1 for Hindi logged to `experiments.csv`; checkpoints saved.

### T-207 — Baseline validation gate ⚠️

Compare the Hindi in-language macro-F1 to IndicFinNLP's published monolingual baseline.
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

`experiments.csv` is the one sanctioned CSV — it is a run log, not corpus data. Every
read or write of corpus data still goes through `src/corpus_io.py` and stays Parquet
(CLAUDE.md §4 rule 2, §5); never use pickle for anything under `data/`.

---

## Working agreement

- Report the baseline gap honestly. Do not tune until it matches — diagnose first.
- If a run OOMs, report the config. Do not silently reduce batch size.
- If a run fails, report it and continue. Do not silently swap the encoder,
  hyperparameters, or split.
- **When a check fails, show the failing rows.** A count is not a diagnosis.
- **Report row-count anomalies immediately.** Do not work around them.
- **The long GPU jobs are run by the project owner, not the assistant.** Finish the
  code, then hand over the exact commands to run.
- Prefer boring code. This must be reproducible cold in October.
- **Commit after every meaningful change.** A completed task, a passing check, a fixed
  bug, or any other self-contained unit of work gets its own commit before moving on —
  don't let unrelated changes pile up uncommitted.
