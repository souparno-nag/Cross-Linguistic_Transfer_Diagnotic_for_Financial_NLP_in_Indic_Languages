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

**Status: DONE.** `scripts/t500_smoke.py`, `tests/test_t500_smoke.py`. Two legs: a
static audit (CPU, instant, no model load) and the end-to-end run. Running the audit
first was worth it — the training and inference path was *not* in fact agnostic, and
it found five blockers:

| blocker | fix |
|---|---|
| `evaluate.run_id_for` pasted `indicbert` into the checkpoint name literally, so `plan()` reported every condition blocked for any other encoder | `Encoder` gained a `slug`; run_ids build from it, threaded through `plan`/`run_condition`/`run_all` |
| prediction log had no `encoder_id` column | added, taken from the checkpoint's own run config |
| prediction-log and failure-set paths had no encoder level | optional `encoder` argument on both; `None` keeps the original path |
| `t302_predictions` wrote the in-language ceiling log to the un-namespaced path whatever the encoder | takes the encoder from the checkpoint it loads |
| `results_tables` read source and target logs from the default path | `encoder` threaded through all five entry points |

The last two were found *after* the sweeps ran and are the dangerous ones: both would
have silently mixed encoders rather than failing. The ceiling one would have
overwritten IndicBERT's log, changing the meaning of every transfer gap already
committed.

**The audit predicts the VRAM wall before any model is loaded.** Under AMP the master
weights stay fp32 and AdamW keeps two fp32 moments per parameter, so a full fine-tune
costs 16 bytes per parameter regardless of batch size. That check runs on the `meta`
device and costs nothing.

**A T-205 bug surfaced here and was fixed** (`src/vram.py`). `GradScaler` skips
`optimizer.step()` when the first scaled gradients overflow, which on a synthetic
batch they reliably do, and AdamW allocates its moment buffers inside that call — so
the committed budget table recorded peaks *below* the encoders' own
weights+gradients+optimiser floors, which is impossible for a completed step. That is
why XLM-R looked like it fitted in 2.76 GiB. The probe now forces the allocation, and
every row carries the floor beside the measured peak so the failure mode is visible in
the artefact.

### T-501 — XLM-R: train and evaluate

Native splits × 3 seeds, then every condition in `eval_conditions.json`.
**Done:** all result rows in `experiments.csv`; per-instance predictions written to
`data/predictions/`. GPU ≈6h.

**Status: NOT RUN — VRAM-blocked, deferred.** XLM-R base cannot be fully
fine-tuned on this project's GPU at any batch size:

| encoder | params | fixed cost (16 B/param) | headroom on 3.68 GiB |
|---|---|---|---|
| `indicbert-v2` | 34M | 0.50 GiB | +3.18 |
| `mbert-base` | 179M | 2.65 GiB | +1.03 |
| `xlm-r-base` | 279M | **4.14 GiB** | **−0.46** |

Confirmed on the card: the run dies allocating 734 MiB, which is XLM-R's
250k-token embedding matrix in fp32. Because this is fixed parameter cost, no
reduction in `batch_size`, `max_len` or gradient checkpointing can change it.
Configs are kept, protocol-identical and under test, in
`configs/train/deferred/` with the arithmetic in its README. T-503 records why
adapters were not used instead.

### T-502 — mBERT: train and evaluate

Same protocol.
**Done:** as above. GPU ≈5h.

**Status: DONE.** Baselines, condition sweeps, ceiling logs and failure sets,
all three seeds, both classification tasks. Everything is filed under
`data/{predictions,failures}/task_{n}/mbert-base/`, leaving IndicBERT's
committed artefacts byte-identical.

| | indicbert-v2 | mbert-base |
|---|---|---|
| task 2 in-language | 0.8226 ± 0.0044 | **0.8742 ± 0.0170** |
| task 3 in-language | 0.1526 ± 0.0249 | **0.3354 ± 0.0417** |

The comparison is on identical rows, not merely comparable ones:
`data.stratified_split` is seeded and encoder-blind, so for a given (task, seed)
both encoders were scored on the same 224 (task 2) or 80 (task 3) test items.
Verified, not assumed.

**One VRAM-forced protocol difference, measured rather than assumed.**
`task2_hin_mbert` uses `batch_size 8 / grad_accum 2` where its IndicBERT
sibling uses 16 / 1. The effective batch is 16 in both, and because the loss is
a mean, two micro-batches of 8 produce the same gradient as one batch of 16 —
optimiser, learning rate, schedule and update count unchanged. mBERT at 16 ×
`max_len` 192 OOMs, reproduced in a clean process, so this is the deviation
hard rule 2 permits. Applied to mBERT alone by decision, since matching it on
IndicBERT would mean re-running Phase 2 and Phase 3 and re-validating the T-207
gate to erase a difference with no expected effect. The parity check in
`tests/test_t500_smoke.py` allows `batch_size`/`grad_accum` to differ **only**
when their product does not, and `reports/baselines.md` shows the effective
batch so the difference is visible where the two encoders first sit side by
side. Task 3 needed no such change.

21 of 63 (condition, seed) pairs are runnable per task, as for IndicBERT; the
other 42 are blocked on Phase 2's deferred Bengali and Telugu baselines and are
reported as blocked, not skipped.

### T-503 — MAD-X adapter fallback (contingency)

**Only if T-501 or T-502 is VRAM-blocked.** Language and task adapters per Pfeiffer
et al. [17], which is compute-feasible on the 4 GB card.
**Done:** adapter run completes, or the task is closed as unnecessary with a note.

**Status: CLOSED as unnecessary.** Full note in `reports/t503_adapter_decision.md`.

The contingency did fire — T-501 is VRAM-blocked outright and T-502 was blocked on
task 2 — but neither needed adapters. T-502's block was an *activation* shortage, so
re-shaping the batch cleared it at an unchanged effective batch. T-501's is fixed
parameter cost, which adapters genuinely would fix by freezing the base model; they
were feasible and rejected anyway, for three reasons:

1. **They would destroy the comparison they were meant to rescue.** IndicBERT and
   mBERT were fully fine-tuned, so an adapter-trained XLM-R would differ in *how it
   was trained* as well as in what it was pretrained on — and a low score could then
   mean capacity dilution or simply less room to adapt, with nothing to separate them.
   That is exactly T-505's question, made strictly less answerable.
2. **Removing that confound costs more than the arm is worth** — all three encoders
   would need re-running under adapters, reopening Phase 2, Phase 3 and the T-207 gate.
3. **The dependency risk is disproportionate.** MAD-X means AdapterHub's `adapters`
   package, which pins `transformers` versions; this project's `transformers==4.57.6`
   is load-bearing for IndicTrans2 and therefore for the frozen corpus (CLAUDE.md §3.2).

### T-504 — Cross-encoder comparison table

All three encoders × all conditions, mean ± std, gaps with confidence intervals.
**Done:** one table; `reports/encoder_comparison.md`.

**Status: DONE.** `src/encoder_comparison.py`, `scripts/t504_comparison.py`, 17 tests.
`reports/encoder_comparison.{md,parquet}` — 24 of 72 (condition, encoder) cells
computed, 48 blocked and rendered as blocked so the matrix keeps its full shape.

**mBERT beats IndicBERT-v2 in all twelve comparable cells**, by +0.15 to +0.37
macro-F1, every one clearing the two encoders' combined seed spread.

**The gaps split the two tasks apart, and that is the interesting part:**

| | IndicBERT gap | mBERT gap |
|---|---|---|
| task 2 | 0.38 – 0.47 | **0.12 – 0.24** |
| task 3 | 0.50 – 0.51 | 0.45 – 0.56 |

On task 2 mBERT both starts higher and loses far less crossing languages. On task 3
the two lose almost exactly the same amount despite mBERT starting three times higher,
and on `hin→mal` mBERT's gap is the *larger*. So "the better encoder transfers better"
holds on task 2 and does not hold on task 3, and should not be reported as a property
of the encoders.

Translationese agrees on direction: reading Hindi machine-translated from another
language costs IndicBERT 5.7–6.9 points on task 2 and 26.2–26.7 on task 3, against
mBERT's 2.4–3.3 and 18.0–18.9.

Three decisions are encoded rather than left to the reader. Confidence intervals are
the **mean of the three seeds' 1000-sample bootstrap intervals**, not a single pooled
interval, and the report says which. `beats noise?` compares each delta against the
combined seed spread rather than leaving it to the eye. XLM-R keeps a row carrying its
exclusion reason, because a dropped row would hide the loss.

**Caveat:** IndicBERT's task-3 gap standard deviations run ±0.17 to ±0.19 against
mBERT's ±0.03 to ±0.07, so task-3 gap comparisons are much softer than task 2's.

### T-505 — Capacity-dilution analysis

Does the Indic-specialised encoder beat the 100-language model, and where? Ties to
Conneau et al. [13], who describe the positive-transfer versus capacity-dilution
trade-off.
**Done:** written analysis with supporting numbers. Report it faithfully if XLM-R wins —
that contradicts the motivation for choosing an Indic encoder and is worth stating.

**Status: DONE.** `src/capacity.py`, `scripts/t505_capacity.py`, 11 tests.
`reports/capacity_dilution.md`. Task numbers are read back from T-504's and T-206's
parquets, never transcribed; the two new measurements are the parameter split and the
tokenisation profile, both CPU-only and weight-free.

**The bet behind choosing an Indic encoder does not pay off** — but the comparison as
run cannot attribute that to multilinguality, and three findings say why.

**The confound is twice as large as the totals suggest.** IndicBERT-v2 is ALBERT-based
and shares one layer's weights across twelve, so while its parameter count is 5×
smaller, its transformer *body* is 7.8M against 85.6M — **11×**. Most of its count is a
200k embedding table.

**The specialisation advantage is real and did not survive into the task.** On the same
sentences in four languages, IndicBERT needs **14–24% fewer tokens** than mBERT, and
the advantage grows with distance from the corpus centre (1.16× Hindi, 1.32×
Malayalam), with no meaningful UNK rate either side. It holds the vocabulary edge it
was chosen for and still loses every task measure.

**The deficit is concentrated in transfer, not spread evenly.**

| task | in-language Δ | cross-lingual Δ | amplification |
|---|---|---|---|
| 2 | +0.052 | +0.268 | **5.2×** |
| 3 | +0.183 | +0.208 | 1.1× |

On task 2 IndicBERT is nearly competitive at home and collapses abroad — its
cross-lingual mean of 0.519 on a *binary* task is closer to "not transferring" than to
"transferring badly". An encoder merely too small would be behind by a similar margin
everywhere. Task 3 shows no amplification and is explicitly discounted: there is little
in-language competence there to lose.

**What that concentration means is left as a labelled hypothesis**, because this design
cannot separate cross-lingual alignment from the 11× body difference. The report names
the comparison that would have settled it: `mbert-base` and `xlm-r-base` have
**identical 85.6M-parameter bodies** and differ almost only in vocabulary and
pretraining breadth — capacity held constant, breadth varied. That is precisely the arm
VRAM blocked, so **T-505's question is recorded as left open** rather than answered by
the two encoders that happened to run.

### T-506 — Emit failure sets for Phase 6

Run the Phase 3 mismatch filter over the new predictions so Phase 6 can label them.
**Done:** `data/failures/{condition}.parquet` written with `encoder_id` populated.
**Coordinate with Phase 6 before running** — they must have finished T-601 first.

**Status: NOT STARTED**, deliberately. The inputs already exist —
`data/failures/task_{n}/mbert-base/` holds the six runnable Hindi-sourced conditions
per task, written by T-307 with the encoder namespaced — so this task is about
handing them over, not producing them. Held pending the coordination the brief
requires.

---

## Outputs

| Artefact | Path |
| --- | --- |
| Run log | `experiments.csv` (append-only) |
| Checkpoints | `checkpoints/task{n}_{lang}_{slug}_seed{s}/` |
| Predictions | `data/predictions/task_{n}/[{encoder}/]{condition}.parquet` |
| Failure sets | `data/failures/task_{n}/[{encoder}/]{condition}.parquet` |
| Comparison | `reports/encoder_comparison.{md,parquet}` |
| Capacity analysis | `reports/capacity_dilution.md` |
| Adapter decision | `reports/t503_adapter_decision.md` |

`encoder_id` must be populated on every row of every artefact.

**As built, two details differ from the paths above as originally written.** Both
levels are load-bearing rather than decorative:

* The `task_{n}` level predates Phase 5 (CLAUDE3.md T-302) — task 2 and task 3 produce
  identically-named conditions, and `item_id` is only unique *within* a task.
* The `{encoder}` level is **omitted for `indicbert-v2`**, whose Phase 3 artefacts were
  committed before Phase 5 and are what `src/diagnostics` currently reads; it globs
  those directories non-recursively, so an encoder subdirectory is invisible to Phase 6
  and their in-flight work is untouched. Every other encoder gets its own level.
  `predictions.log_path(..., encoder=None)` is that default.

One consequence, recorded rather than fixed: IndicBERT's Phase 3 logs predate the
`encoder_id` column and still lack it. Its encoder is recoverable from `run_id` and
from `experiments.csv`, so nothing is ambiguous, but backfilling would rewrite files
Phase 6 reads and is left for a deliberate migration.

---

## Working agreement

- **This phase is deferrable.** If time runs short, a single-encoder project is complete
  and defensible. Do not let it consume Phase 6's GPU budget.
- **Report a lost comparison honestly.** If mBERT cannot be fine-tuned within VRAM and
  adapters were used instead, that is a protocol difference and must be stated, not
  smoothed over.
- Prefer boring code. This must be reproducible cold in October.
