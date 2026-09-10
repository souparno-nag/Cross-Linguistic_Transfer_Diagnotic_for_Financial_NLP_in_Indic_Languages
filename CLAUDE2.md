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

The repo ships **no `tokenizer.json`**, so `AutoTokenizer` fetches `spiece.model`
(5.6 MB SentencePiece) on first use — and that fetch stalls at zero bytes on this
machine, the §3.1 HF-stall symptom again. Pre-warm the cache once, outside pytest,
before running the `slow` suite:

```
hf download ai4bharat/indic-bert config.json spiece.model spiece.vocab
```

It may `Read timed out` mid-file and resume itself; let it finish. `mBERT` and `XLM-R`
download without trouble.

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
stratified split) plus 13 `slow` tests tokenising all 4 languages × 3 encoders, all
passing once `indicbert-v2`'s gated repo is accepted and its tokeniser pre-fetched (see
Encoders).

### T-203 — Training loop
`src/train.py`, `src/metrics.py`, `scripts/t203_smoke.py`, `tests/test_train.py`

Linear head, cross-entropy, AdamW, warmup, early stopping on dev macro-F1.

`Classifier` is the whole architecture: encoder → `last_hidden_state[:, 0]` (the
`[CLS]` position, **not** the BERT pooler's extra dense+tanh) → one `nn.Linear`. AdamW
with the standard no-decay group for biases and LayerNorm; linear warmup + decay over
`steps_per_epoch * epochs`. `train()` checkpoints atomically after every epoch and
resumes from that checkpoint on restart (intermittent GPU, CLAUDE.md §3); a checkpoint
written for a different config hash is refused. Early stopping restores the best-dev
weights before returning. `src/metrics.py` owns accuracy + macro-F1 + per-class F1 so
T-206/T-207 compute them identically.

**Determinism (`hard rule 5`) needed a fix found by the test:** the classifier head is
randomly initialised, so `build_model()` must seed *before* constructing it — otherwise
two runs of one config start from different head weights and diverge from epoch 0. With
that, two `overfit_subset` runs produce byte-identical history.

**Done.** `python -m scripts.t203_smoke --device cuda`: mBERT on 48 class-balanced
Hindi rows (task 2), 30 epochs, **final train macro-F1 1.0000** (need > 0.95) — the loop
learns, the wiring is right. 15 fast tests + 6 `slow` (overfit, run-to-run determinism,
checkpoint resume through a simulated interruption, config-mismatch refusal). The subset
is class-balanced because macro-F1 on an all-one-class sample is capped below 1.0.

*On the 3050 the 48-row run repeatedly logged `expandable_segments: memory mapping
failed with OOM` yet completed — the card was near-full from the desktop session. Real
training sizes are T-205's problem; note it there.*

### T-204 — Config system
`src/config.py`, `src/experiments.py`, `configs/train/*.yaml`, `tests/test_config.py`

YAML: seed, LR, batch size, max_len, epochs, encoder, split. Config hash written into
every result row.

One YAML per baseline (`configs/train/task{2,3}_hin_indicbert.yaml`). `load_run_config()`
reads it into a frozen `RunConfig` and applies keyword overrides — T-206 sweeps seeds by
overriding, not by keeping three near-identical files. It coerces YAML's type surprises
(`2e-5` parses as a *string* in PyYAML) and rejects unknown keys rather than ignoring
them. `RunConfig` is a superset of `train.TrainConfig`: a metric depends on the data
split and the dev/test fractions as well as the hyperparameters, so `RunConfig.hash()`
covers all of it, over the *resolved* config so `block: null` and `block: H` for Hindi
hash equal. `run_training()` is the load→split→train→evaluate unit T-206 repeats per
seed. `src/experiments.py` appends rows to `experiments.csv` — the one sanctioned CSV
(a run log, not corpus data) — with the fixed columns `run_id, encoder, split, seed,
config_hash, accuracy, macro_f1, date`.

**Done.** 21 fast tests (loader coercion, unknown-key rejection, validation, hash
sensitivity, CSV round-trip) + the `slow` acceptance: two `run_training` calls on one
config produce identical `accuracy`, `macro_f1` and epoch history.

### T-205 — VRAM budgets
`src/vram.py`, `scripts/t205_vram.py`, `tests/test_vram.py`

Max batch × max_len per encoder, with fp16 and gradient accumulation. Set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before torch is imported — on the
4 GB card, without it training OOMs on fragmentation rather than genuine exhaustion
(CLAUDE.md §3.3). `src/train.py` sets it at module scope.

Observed in T-203: even mBERT (711 MB) on 48 rows at `batch 16 / max_len 128` drove the
3050 to a few MB free and logged `expandable_segments: memory mapping failed with OOM`
on nearly every step, recovering each time. The desktop session holds ~1 GB, so the
real budget is ~3 GB, not 4. Close every other GPU user before a run, expect `fp16` and
`grad_accum` to be mandatory not optional, and treat a laptop reboot as part of the
setup if the driver wedges (CLAUDE.md §3.5).

`src/vram.probe_step` runs one real forward → backward → AdamW step at a given
`(encoder, batch, max_len)` with `attention_mask` all ones (worst case) and reports
whether it fit and its `max_memory_allocated` peak; `largest_batch` / `budget_table`
sweep the grid. `scripts/t205_vram.py` writes `reports/vram_budget.{md,parquet}` and,
with `--full-epoch`, trains one epoch on each shipped config's native fold to prove no
OOM across an epoch — exiting non-zero if any config OOMs.

**Done.** `python -m scripts.t205_vram --full-epoch` on the 3050 (3.7 GiB visible
total). Worst-case probe, fp16, all-ones mask — largest batch that fit / peak GiB:

| encoder | len 64 | len 128 | len 256 |
|---|---|---|---|
| indicbert-v2 | 64+ / 1.60 | 64+ / 3.04 | 32 / 3.04 |
| xlm-r-base   | 64+ / 2.76 | 32 / 2.76  | 12 / 2.37 |
| mbert-base   | 64+ / 2.38 | 48 / 3.32  | 24 / 3.16 |

(`64+` = the probe's ceiling, not a limit.) The ceiling in practice is ~3.3 GiB; batch
48–64 at `max_len 128` is the edge for the BERT-sized encoders.

**Both shipped configs pass with wide headroom** — `task2_hin_indicbert` peaked at
**1.14 GiB** (29 s/epoch), `task3` at **0.73 GiB** (7 s). Real training uses per-batch
dynamic padding and mostly-short text, so it costs a fraction of the all-max-length
worst case. `batch_size 16` in the YAMLs stands; `fp16: true` matters, `grad_accum`
is not needed for IndicBERT-v2 on either task.

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
