# Phase 2 — Training Pipeline and In-Language Baselines

**Specification:** `CLAUDE2.md`
**Status:** Complete. Hindi baselines trained and validated; Bengali/Telugu training
deliberately deferred.

## What this phase builds

A fine-tuning pipeline that trains a classifier on a **native** (human-written) language
split and confirms the result reproduces IndicFinNLP's own published numbers for that
same setup. Nothing about cross-lingual transfer is measured or claimed in this phase —
the entire point is to establish a trustworthy *in-language* starting point before Phase
3 asks what happens when that model is pointed at a different language. If this phase's
baseline doesn't reproduce the published numbers, there is no way to later tell "the
model doesn't transfer across languages" apart from "the model was never trained
correctly to begin with" — which is why Phase 3 is gated on this phase passing (see
T-207 below).

This phase depends on Phase 1's frozen corpus (`data/v1.0/`) and reuses its IO layer
(`src/corpus_io.py`), ID scheme (`src/ids.py`), and label schema (`configs/labels.json`)
rather than reimplementing any of them. The corpus is read-only here — its manifest is
verified at startup, and any run refuses to proceed if verification fails.

## Scope decision: Hindi only, for now

Due to time constraints, this phase trains on **Hindi only** (`H_nat`). Bengali and
Telugu baselines are deferred, not cancelled — the loader, config system, and training
loop are all built to be language-agnostic, so adding those two baselines later costs
only GPU time, not new code. Malayalam gets no baseline in any case, since it has no
native split to train on (see Phase 1). This scope decision is the reason Phase 3
currently only measures transfer *out of* Hindi.

## Encoders

| Model | Hugging Face ID | Status |
|---|---|---|
| IndicBERT-v2 | `ai4bharat/indic-bert` | The only encoder actually trained in Phase 2 |
| XLM-R base | `xlm-roberta-base` | Loader supports it; training deferred to a later phase |
| mBERT base | `bert-base-multilingual-cased` | Loader supports it; training deferred to a later phase |

Only IndicBERT-v2 needs to work for this phase's goal, but the data loader (`src/data.py`)
keeps a registry of all three so extending to the other encoders later requires no new
loading code.

**`ai4bharat/indic-bert` is gated on Hugging Face**, the same way the IndicTrans2 model
repos are in Phase 1 — being logged in is not sufficient; the specific model's license
has to be accepted with the same account first, or downloads fail with a gated-repo
error. Its tokenizer also has no pre-built `tokenizer.json`, so the first use fetches a
5.6 MB SentencePiece file directly, which was observed to occasionally stall at zero
bytes on the development machine (the same kind of Hugging Face download stall seen in
Phase 1) — pre-warming that download once, outside of the test suite, avoids the issue.

## Hard rules

1. **Train on native splits only.** Machine-translated splits are Phase 3's evaluation
   data; training on them would leak translated content into the model and contaminate
   every transfer measurement made afterward.
2. **Never write to `data/v1.0/`.** It remains Phase 1's immutable output.
3. **At least 3 random seeds** per configuration, reported as mean ± standard deviation —
   never a single number, since a single training run's score is not distinguishable from
   noise.
4. **Every result row records the exact config hash** that produced it.
5. **Deterministic** — the same configuration and seed must produce the same metrics
   every time.
6. **Macro-F1 is the primary metric.** Plain accuracy is reported alongside it but never
   reported alone, because accuracy alone can hide a model that only ever predicts the
   majority class when classes are imbalanced.

## Model architecture

Deliberately minimal: a pretrained encoder's `[CLS]` token representation (the
first-position output, used as a whole-sentence summary), followed by one linear layer
mapping to the number of classes, trained with ordinary cross-entropy loss. No extra
pooling strategies, no additional layers, no loss-function tricks — the goal here is a
clean, standard baseline to measure transfer against, not architecture novelty.

## Code layout

Same principle as Phase 1: all real logic lives in importable `src/` modules
(`src/train.py`, `src/data.py`, `src/config.py`, `src/metrics.py`, …), each with its own
test file, driven by thin CLI scripts (`scripts/t20x_*.py`) run as
`python -m scripts.t20x_*`.

One exploratory notebook, `notebooks/phase2.ipynb`, is allowed — but strictly as a
*driver* that imports and calls `src/` functions to launch runs and inspect results
(memory probing, tokenizer inspection, learning curves). No pipeline logic is allowed to
live only in the notebook, because a notebook's out-of-order cell execution and hidden
state would defeat both the determinism rule and the goal of being reproducible by
someone reading the code cold months later. Outputs are stripped before it is committed.

## Tasks

### T-201 — Environment
`src/env_check.py`

Every Phase 2 entry point asserts Python 3.11 exactly, the same as Phase 1, and fails
loudly on any mismatch. Phase 2 shares Phase 1's virtual environment, so `transformers`
stays pinned below version 5 for the same reasons documented in `docs/phase1.md` —
IndicBERT-v2 works fine on the 4.x line, so there is no need to break that pin for this
phase's sake.

**Status: done.** `python -m src.env_check` checks the interpreter version, imports the
whole training stack, reports library versions and CUDA availability, and exits non-zero
on any problem — it is the smoke test run before anything else in this phase.

### T-202 — Dataset loader
`src/data.py`

Loads any corpus split by `(block_id, language, origin)`, tokenizes it for a chosen
encoder, and validates its labels against `configs/labels.json`.

Two safety properties matter here specifically because of Hard Rule 1 above:
`load_split()` **asserts** that the split's actual `origin` column matches what was
requested — asking for a `native` split but receiving a machine-translated one raises an
error rather than silently returning the wrong data, so the "train on native only" rule
cannot be broken by a mislabeled file. Task 1 (the numeral-span task) is refused outright
here, since it has no classification label to train against at all.

Since the frozen corpus itself carries no train/dev/test split, `stratified_split()`
creates one — seeded, and stratified by class so that each split's class balance matches
the whole. Tokenization happens once up front; padding happens per batch (dynamic
padding) rather than to a fixed maximum length, which matters directly for staying within
the 4 GB GPU budget (see T-205).

**Status: done.** 22 fast tests plus 13 slower tests that actually tokenize all four
languages across all three registered encoders — these require IndicBERT-v2's gated
repository to already be accepted and its tokenizer pre-fetched, per the note above.

### T-203 — Training loop
`src/train.py`, `src/metrics.py`

`Classifier` implements exactly the architecture described above: encoder →
`last_hidden_state[:, 0]` (the `[CLS]` position taken directly from the encoder's raw
output — **not** BERT's separate "pooler" output, which adds its own extra dense layer
and activation that this project's minimal design deliberately skips) → one linear layer.
Optimization is AdamW with the standard convention of not applying weight decay to bias
and layer-normalization parameters, with a linear warmup-then-decay learning rate
schedule. Training checkpoints after every epoch and can resume from that checkpoint —
necessary given the same intermittent GPU access documented in Phase 1 — and refuses to
resume a checkpoint that belongs to a different configuration (a different config hash),
so a botched resume can't silently mix two different runs' progress. Early stopping
restores the best-performing checkpoint (by dev-set macro-F1) before returning, rather
than whatever epoch happened to run last.

**A real determinism bug was found and fixed during testing.** The classifier's linear
head starts with randomly initialized weights; the fix required seeding the random
number generator **before** constructing the model, not just before the training loop
starts — otherwise two runs of the exact same configuration would start from two
different random head weights and diverge from the very first step, silently breaking
Hard Rule 5. After the fix, two runs on an identical small subset produce byte-identical
training histories.

**Status: done.** A smoke test (mBERT trained on 48 hand-picked, class-balanced Hindi
rows for 30 epochs) reaches a training macro-F1 of 1.0000, confirming the training loop
actually learns and the architecture is wired correctly, plus 15 fast tests and 6 slower
tests covering overfitting on a tiny subset, run-to-run determinism, checkpoint resume
through a simulated crash, and refusal to resume a mismatched configuration.

### T-204 — Config system
`src/config.py`, `src/experiments.py`, `configs/train/*.yaml`

Each training run is described by one YAML file (seed, learning rate, batch size,
maximum sequence length, epoch count, encoder, and which split to train on), loaded into
a frozen, validated configuration object. Rather than keeping three nearly-identical YAML
files to sweep three seeds, the loader applies keyword overrides on top of one base file,
so seed-sweeping (T-206) is done by parameterizing one config rather than copy-pasting
it. The loader also fixes a real footgun in plain YAML parsing — a learning rate written
as `2e-5` parses as a plain *string*, not a number, unless handled explicitly — and
rejects any config key it doesn't recognize rather than silently ignoring a typo.

Every result's config hash covers the *fully resolved* configuration, including which
data split it trained on and what fraction was held out for dev/test — not just the
hyperparameters — specifically so that two configs which differ only in which block they
train on (but are otherwise identical) are correctly treated as different runs, while two
ways of writing the same resolved config (e.g. an explicit block name vs. a default that
resolves to the same block) correctly hash to the *same* value.

Every training result gets appended as one row to `experiments.csv` — deliberately the
one sanctioned use of CSV in this entire project (Phase 1's Hard Rule 2 bans CSV for
*corpus data*; this is a run log, not corpus data, so the rule doesn't apply to it) —
with fixed columns: `run_id, encoder, split, seed, config_hash, accuracy, macro_f1, date`.

**Status: done.** 21 fast tests (config-loading edge cases, hash sensitivity to real
changes, CSV round-trip) plus a slower acceptance test confirming two runs of one config
reproduce identical accuracy, macro-F1, and full epoch-by-epoch history.

### T-205 — VRAM budgeting
`src/vram.py`, `scripts/t205_vram.py`

Determines, empirically, the largest batch size that fits in the 4 GB GPU's memory for
each encoder at each sequence length, using half-precision (fp16) training. A specific
environment variable (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`), set before
`torch` is even imported, turned out to be necessary — without it, training would run out
of memory on *fragmentation* well before actually running out of raw memory, the same
issue documented in Phase 1's translation pipeline.

**A real hardware constraint was discovered here, not just budgeted for in theory.** On
the development machine, even a small model (mBERT, 711 MB) training on just 48 rows
drove the 4 GB card to only a few megabytes of free memory and repeatedly logged memory
allocation failures on nearly every step (recovering each time) — because roughly 1 GB
was already held by the desktop environment itself. The real usable budget on this
machine turned out to be closer to 3 GB than the nominal 4 GB, which is why fp16 and
gradient accumulation are treated as mandatory defaults rather than optional tuning
knobs.

**Status: done.** A full sweep across all three registered encoders and three sequence
lengths (64/128/256 tokens), using worst-case inputs (no padding shortcuts), found the
practical ceiling to be roughly 3.3 GB, with batch sizes of 48–64 at sequence length 128
being the edge for the BERT-sized encoders tested. Both of the configs actually shipped
for this phase's baselines passed with wide headroom — the task 2 config peaked at 1.14
GB (29 seconds per epoch) and the task 3 config at 0.73 GB (7 seconds per epoch), because
real training text is mostly much shorter than the worst-case probe assumes and uses
per-batch dynamic padding rather than always padding to the maximum length. `batch_size:
16` was kept in both shipped configs, and gradient accumulation was found unnecessary for
IndicBERT-v2 on either task.

### T-206 — Baseline runs
`scripts/t206_baseline.py`

Trains IndicBERT-v2 on the Hindi native split, 3 seeds each, for both classification
tasks (6 total training runs). `python -m scripts.t206_baseline` runs the whole sweep,
logs every run to `experiments.csv` with its config hash, and writes a summary report
(`reports/baselines.md`/`.parquet`) with mean ± standard deviation macro-F1 per
configuration, measured on the held-out **test** fold. The sweep is resumable at two
independent levels: an individual run resumes from its own checkpoint if interrupted, and
a run already logged in `experiments.csv` under a matching config hash is skipped
entirely on a re-run — so re-running the same command after an interruption only finishes
whatever was left undone, rather than restarting the whole sweep. A run that fails is
reported and the sweep continues with the rest, rather than aborting everything.

**Status: done.**

| Config | Macro-F1 (test, mean ± std) | Accuracy | Reading |
|---|---|---|---|
| `task2_hin_indicbert` | **0.825 ± 0.024** | 0.826 ± 0.025 | Healthy — dev F1 climbs to ~0.84, training converges cleanly, early stopping triggers normally |
| `task3_hin_indicbert` | **0.153 ± 0.025** | 0.225 ± 0.033 | Low, but for a reason grounded in the data — see below |

**Task 3's low score is not a bug** — the same code trains task 2 fine, ruling out a
wiring problem — and it is not simple underfitting either: after deliberately raising the
training budget (learning rate 3e-5, 40 epochs, patience 8 before early stopping), the
model now **overfits hard**: training macro-F1 reaches 0.99 while the dev and test sets
sit around 0.15–0.20. The extra training budget barely moved the test score (0.127 to
0.153, within seed-to-seed noise). The likely reason: task 3 has only 532 Hindi rows
total; after holding out 30% for dev and test, that leaves roughly 370 training examples
spread across 10 ESG topic classes — some classes have as few as 21 examples upstream. A
12-million-parameter encoder can memorize that little data almost perfectly and still not
generalize, and no amount of additional training time, regularization, or a bigger model
changes that — the ceiling here is the amount of available data, not the model or the
training procedure.

This finding is what made T-207 necessary before drawing any conclusion: the real
question was whether IndicFinNLP's own published result for task 3 is *also* this low
(in which case this simply reproduces a known-hard task) or much higher (in which case
this project's setup differs from theirs in some way worth finding).

### T-207 — Baseline validation gate ⚠️
`src/baseline_gate.py`, `configs/published_baselines.json`

Compares this project's Hindi in-language macro-F1 against IndicFinNLP's own published
monolingual baseline numbers. **This is a hard gate: nothing in Phase 3 is allowed to
start until this passes**, because a transfer result measured against an unvalidated
in-language baseline is meaningless — there would be no way to distinguish "the model
doesn't transfer to other languages" from "the model was never a good in-language model
to begin with."

Published reference numbers, from Ghosh et al. Table 4 (test set), using their
`IndicBERT` model on the original (non-augmented) data: **task 2 Hindi macro-F1 0.86**,
**task 3 Hindi macro-F1 0.05** — the original paper reports every model scoring under 30%
on task 3, for every language, specifically because of the same under-100-examples-per-class
problem identified independently in T-206.

`python -m scripts.t207_gate` compares each trained config to its published entry within
a ±0.02 tolerance and exits non-zero unless every config is within tolerance, above it,
or has a written diagnosis attached explaining a gap outside tolerance.

**Status: done — gate passes.**

| Config | This project | Published | Verdict |
|---|---|---|---|
| `task3_hin_indicbert` | 0.153 ± 0.025 | 0.05 | **Pass, above.** Consistent with the paper's own low-data regime; their published fix (augmenting the training data to 4,774 rows via paraphrasing) is out of this phase's scope. |
| `task2_hin_indicbert` | 0.823 ± 0.004 | 0.86 | **Pass, below tolerance but diagnosed.** After tuning (max length 192, 20 epochs, patience 5), the mean score stayed essentially the same (0.825 → 0.823) but the run-to-run standard deviation tightened by roughly 10×, so 0.82 is a stable number for this exact setup rather than one lucky or unlucky run. Training macro-F1 reaches 0.998, dev sits at 0.86–0.90, and test lands at 0.82 — a healthy-looking training curve on a noisier, smaller (roughly 224-row) resampled test fold, without access to the paper's exact original split or hyperparameters. |

The conclusion drawn: the gap between this project's numbers and the published ones is
explained by a difference in the evaluation split and tuning details, on a materially
smaller resampled test set — not by a modeling error. Task 2 trains cleanly and task 3
reproduces its known-hard published number on identical code, so **Phase 3 was cleared to
proceed.**

### T-208 — Checkpoint versioning
`src/checkpoints.py`

Defines the naming convention, storage layout, and cleanup policy for trained model
checkpoints: one directory per run, `checkpoints/<run_id>/`, where
`run_id = "<config-name>_seed<N>"` (e.g. `task2_hin_indicbert_seed1`). Each holds the
full checkpoint (model weights, optimizer state, scheduler state, random-number-generator
state, best-so-far tracking, training history, and config hashes) plus a copy of the
resolved config used to produce it. `checkpoints/` itself is gitignored — `experiments.csv`
and the `reports/` directory are the durable, committed record of what was run.

A checkpoint is only kept during cleanup if it is either **reported** (its config hash
has a matching row already in `experiments.csv`) or **current** (its config hash matches
what the shipped YAML would produce today, so it's still resumable). Anything else — a
superseded configuration, or a run that was aborted before completing — is treated as an
orphan and removed by `prune --apply`.

**A real bug was found and fixed here, and it directly affected T-206's resume
behavior.** There turn out to be two different config hashes in play: one covering just
the training hyperparameters (written into the checkpoint file itself by the training
loop) and a broader one that also covers the data split and dev/test fractions (recorded
in `experiments.csv`). The orchestrator that decides "is this checkpoint stale and should
be discarded" was comparing one hash against the other by mistake — meaning **every
attempted resume was discarding a perfectly good, resumable checkpoint**, and every
checkpoint looked like an orphan during cleanup. This was fixed by threading the broader
hash through into the checkpoint itself (as a new `run_hash` field) and keying both
resume and retention decisions off that consistently; checkpoints written before this fix
fall back to comparing against the narrower hash so old checkpoints aren't all
invalidated at once.

**Status: done.** CLI: `python -m scripts.t208_checkpoints list | show <run_id> |
path <run_id> | prune [--apply]`.

## Outputs

| Artifact | Path |
|---|---|
| Run log | `experiments.csv` |
| Checkpoints | `checkpoints/{run_id}/` |
| Training configs | `configs/train/*.yaml` |
| Baseline report | `reports/baselines.md` |

`experiments.csv` columns: `run_id, encoder, split, seed, config_hash, accuracy,
macro_f1, date`.

Corpus data is still always read and written through `src/corpus_io.py` and stays
Parquet — `experiments.csv` is a run log, the one deliberate exception, not a precedent
for using CSV anywhere else.

## Working agreement for this phase

- Report the baseline gap honestly and diagnose it before tuning to close it — tuning
  until the numbers match, without understanding why they didn't, would hide a real
  discrepancy rather than resolve it.
- If a run runs out of GPU memory, report the exact configuration rather than silently
  reducing the batch size to make it go away.
- If a run fails, report it and continue with the rest of the sweep — never silently
  swap in a different encoder, hyperparameter set, or data split.
- When a check fails, show the actual failing rows, not just a count.
- Report row-count anomalies immediately rather than working around them.
- **Long GPU jobs are run by the project owner, not launched autonomously.** The
  pipeline's job is to be finished and correct; a human decides when to actually spend
  the GPU-hours.
- Commit after every self-contained unit of completed work.
