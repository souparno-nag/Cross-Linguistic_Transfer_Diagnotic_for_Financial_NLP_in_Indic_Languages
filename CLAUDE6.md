# CLAUDE.md — Phase 4: Bengali & Telugu Source Training + Zero-Shot Evaluation

Scope: **Phase 4 only.** Train on the Bengali and Telugu native splits and evaluate
zero-shot across their parallel targets, for both encoders this card can fine-tune
(IndicBERT-v2 and mBERT; XLM-R is deferred — see the grid). Do **not** build or
modify diagnostic modules (Phase 6) or aggregate root-cause findings (Phase 7). If a
request drifts there, say so and stop.

---

## What this phase completes

Phase 2/3 covered **block H** (Hindi source). Phase 5 added mBERT for block H and
deferred XLM-R. Phase 4 fills the rest of the grid:

| | IndicBERT-v2 | mBERT | ~~XLM-R~~ |
| --- | --- | --- | --- |
| **Block H** (Hindi source) | Phase 2/3 | Phase 5 | deferred, T-501 |
| **Block B** (Bengali source) | **Phase 4** | **Phase 4** | deferred |
| **Block T** (Telugu source) | **Phase 4** | **Phase 4** | deferred |

Phase 4 owns **4 of the 6 reachable cells**.

### XLM-R is deferred, not scheduled — corrected after reading Phase 5

This spec as first written gave Phase 4 six cells across three encoders. **XLM-R
cannot be fine-tuned on this project's GPU at any batch size**, so its column is
struck out above and Phase 4 is four cells, not six.

The arithmetic is in `configs/train/deferred/README.md` and is not a tuning problem.
Under mixed precision the master weights stay fp32 and AdamW keeps two fp32 moments
per parameter, so a full fine-tune costs 16 bytes per parameter *before a single
sentence is loaded*:

| encoder | params | fixed cost | headroom on a 3.68 GiB card |
|---|---|---|---|
| `indicbert-v2` | 34M | 0.50 GiB | +3.18 |
| `mbert-base` | 179M | 2.65 GiB | +1.03 |
| `xlm-r-base` | 279M | **4.14 GiB** | **−0.46** |

Because that is fixed parameter cost rather than activation cost, no reduction in
`batch_size`, `max_len` or gradient checkpointing changes it — it does not fit at
batch size 1 with a single token. T-501 confirmed this on the card (the run dies
allocating 734 MiB, XLM-R's 250k-token embedding matrix in fp32), and T-503
considered and rejected adapters as the way around it, because an adapter-trained
XLM-R would differ in *how it was trained* as well as in what it was pretrained on
and would make T-505's question strictly less answerable.

Nothing here is a new decision; it is Phase 5's decision applied to the two Phase 4
cells that inherit the same hardware limit. Hindi's XLM-R configs stay under
`configs/train/deferred/`; Bengali and Telugu ones are **not** written, because a
deferred config exists to record work that was done, and this cell's work was never
started.

**What this costs the project is worth stating plainly.** XLM-R and mBERT have
identical-sized transformer bodies (85.6M parameters each) and differ mainly in
vocabulary and pretraining breadth, which made that pair the one controlled test of
whether breadth or capacity explains mBERT beating IndicBERT-v2. T-505 already
records that question as open. Phase 4 does not reopen it and must not imply the
three-encoder comparison exists.

### Two knock-on effects to confirm before starting

- **Phase 2's scope narrows.** Its baseline task named `H_nat`, `B_nat` and `T_nat`.
  Under this split, Phase 2 delivers `H_nat` only; `B_nat` and `T_nat` move here.
  Confirm this is the intent before running anything.
- **Review 3 delivery has lost its slot.** It was the old Phase 4. It still has to
  happen on 16 September and now sits outside the numbered phases — decide where it
  lives.

---

## GPU budget — measured, and far smaller than first estimated

This spec first budgeted "six encoder×block cells × 3 seeds ≈ **25–30 GPU-hours**",
roughly three weeks of lab access. Two things make that wrong, and both were checked
rather than assumed:

1. **There are four cells, not six** — XLM-R is deferred (above).
2. **A run takes minutes, not hours.** `reports/vram_epochs.parquet` measured 31 s
   per epoch for task 2 and ~3.5 s for task 3 on IndicBERT-v2, and the wall-clock
   gaps between consecutive seeds finishing in `experiments.csv` agree (11 epochs in
   5m04s, 16 in 9m10s). Block H's three IndicBERT seeds ran 19/11/16 epochs on task 2
   and 40/24/40 on task 3 under early stopping.

| | per cell (both tasks, 3 seeds) | four cells |
|---|---|---|
| training | ~30 min IndicBERT-v2, ~45 min mBERT | ~2.5 h |
| condition sweeps | ~10–40 min | ~1.5 h |
| **total** | | **~4 GPU-hours** |

So Phase 4 is an afternoon, not three weeks. The estimate held against the whole of
Phase 5, whose mBERT cell ran three seeds in 21 minutes.

**Baseline-first still stands, for a different reason.** It is no longer about
rationing the card; it is that a cell which fails should fail after 30 minutes rather
than after the whole grid has run on a broken assumption. Complete `B_nat` ×
IndicBERT-v2 end to end before starting any other cell.

Cell order:

```md
1. B_nat × IndicBERT-v2     ← prove the cell works end to end
2. T_nat × IndicBERT-v2     ← completes the IndicBERT row across all blocks
3. B_nat × mBERT
4. T_nat × mBERT
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

**Status: DONE, and nothing was hardcoded to block H.** `scripts/t401_smoke.py`,
`tests/test_t401_smoke.py`, 38 tests, plus the eight Phase 4 configs (tasks 2 and 3 ×
Bengali and Telugu × IndicBERT-v2 and mBERT), each a field-for-field copy of its
block-H sibling with only `lang` changed.

The pipeline really was block-agnostic: `RunConfig` derives the block from `lang`,
`evaluate.run_id_for` builds checkpoint names generically, and the prediction,
failure-set and results-table code all take the block from the run config or the
condition. The audit passes on all eight configs with no GPU, and each one names the
seven conditions it unblocks — 14 per task, which closes the 42 (condition, seed)
pairs Phase 3 reported as blocked.

**One guard was missing, and it is the one hard rule 1 turns on.** T-500's parity
check compares a config against the same-*language* IndicBERT one, so it has nothing
to say about `task2_ben_indicbert.yaml` — that config *is* the IndicBERT one for its
language, and the check passes by default having compared it against nothing. A
Bengali baseline could therefore have carried a learning rate Hindi never used, and
T-408's table would have reported a training-recipe difference as a source-language
difference. `check_block_parity` compares against the same-*encoder* Hindi config and
permits only `lang` to move.

It is deliberately stricter than the encoder rule. T-500 allows a VRAM-forced
re-shaping of `batch_size`/`grad_accum` because a bigger model genuinely does not
fit; changing the source language changes neither the model nor the sequence length,
so nothing about a block can force one, and a config that needed it would be tuning.
A test asserts the two rules disagree on exactly that case.

Leg B on the card: block B, 200-row subset, one epoch, reload frozen, predict
`B/hin` — 200 items in, 200 predictions out, no nulls. Its macro-F1 of 0.31 is one
epoch of warmup on 160 rows and is not a quality signal; T-203's overfit test is what
establishes that the loop learns.

### T-402 — Bengali baseline ⚠️ GATE

Fine-tune IndicBERT-v2 on `B_nat`, 3 seeds, evaluate in-language.
**Done:** macro-F1 within ±2 of IndicFinNLP's published Bengali monolingual baseline, or
a written diagnosis of the gap. **No Bengali transfer claim proceeds until this passes.**
GPU ≈30 min for both tasks at 3 seeds, measured — not the 4h first budgeted.

**Blocked on a number this repo does not have.** `configs/published_baselines.json`
carries only Ghosh et al.'s Hindi values. Their Bengali figures (Table 4, rows `2 B IB`
and `3 B IB`) must be added before this gate can be scored at all — the runs can go
ahead, but "within ±2 of published" has nothing to compare against until then.

### T-403 — Telugu baseline ⚠️ GATE

Same for `T_nat`.
**Done:** within ±2 of the published Telugu baseline, or a diagnosis. GPU ≈30 min.
Blocked on the published Telugu numbers (`2 T IB`, `3 T IB`) in the same way T-402 is
blocked on the Bengali ones.

### T-404 — Bengali-source zero-shot evaluation

IndicBERT-v2 → Hindi, Malayalam, Telugu, plus the in-language ceiling and the Bengali
translationese conditions (`B_nat` vs `Bn←H` vs `Bn←T`).
**Done:** all conditions logged; per-instance predictions written.

**Run the sweeps with `HF_HUB_OFFLINE=1`.** `transformers` 4.57.6 makes a live
Hub request inside every `AutoTokenizer.from_pretrained` for a non-local repo id
(`_patch_mistral_regex` calls `model_info` to ask whether the repo is a base Mistral
model), and it does so even when every file is cached. A dropped connection there
killed a 42-pair sweep three conditions in. `data.get_tokenizer` is now cached per
model id, which removes ~70 of those requests per sweep, and `run_all` reports a
failed pair and continues instead of aborting; the environment variable closes the
last one, because `is_offline_mode()` short-circuits the same hook.

**The Mistral warning that appears under that flag is spurious — verified, not
assumed.** Offline, `_is_local` is forced true, so the check enters a branch it
otherwise skips; `ai4bharat/indic-bert`'s `config.json` has `model_type: albert` but
carries **no `transformers_version`**, so `transformers` cannot take its fast path to
rule the model out, sets `mistral_config_detected` and warns "This will lead to
incorrect tokenization". It does not: the warning branch only sets a bookkeeping
attribute, and the branch that actually rewrites the pre-tokenizer regex needs
`fix_mistral_regex=True`, which nothing here passes.

Checked two independent ways, because a warning about *incorrect tokenization* is not
something to wave away by reading the source:

| check | online | offline | result |
|---|---|---|---|
| token ids, 300 real Hindi sentences | 18418 tokens | 18418 tokens | **identical ids** |
| `transfer_hin_to_ben`, 3294 predictions | — | — | **0 differ, max probability delta 0.0** |

The only thing that changes is the value of `tokenizer.fix_mistral_regex` — `unset`
online, `False` offline. The corpus artefacts produced with and without the flag are
byte-comparable, so `hard rule 1`'s identical protocol holds across it.

### T-405 — Telugu-source zero-shot evaluation

Same for block T, including `Te→Ml`.
**Done:** as above.

### T-406 — Extend to mBERT

Repeat T-402 through T-405 for mBERT, in the cell order above. **XLM-R is not part of
this task** — it cannot be fine-tuned on this card at any batch size (see the grid
above), so "the remaining two encoders" as first written is one encoder.

**Done:** every completed cell has baseline + full evaluation logged. Partial completion
is acceptable and must be recorded as such — state which cells ran, not just which
succeeded, and state XLM-R as deferred-on-hardware rather than as not-yet-run.
GPU ≈1.5h for both mBERT cells, measured — not the 17h first budgeted.

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

**As built, the prediction and failure paths carry two more levels than shown above**,
and both are load-bearing rather than decorative — see CLAUDE5.md's Outputs note:

    data/predictions/task_{n}/[{encoder}/]{condition}.parquet
    data/failures/task_{n}/[{encoder}/]{condition}.parquet

The `task_{n}` level exists because tasks 2 and 3 produce identically-named conditions
and `item_id` is unique only within a task. The `{encoder}` level is **omitted for
`indicbert-v2`**, whose Phase 3 artefacts were committed before Phase 5 and are what
`src/diagnostics` reads today; every other encoder gets its own level. Writing to the
paths as first listed above would collide two tasks' conditions in one file.

**One ambiguity to settle before T-408.** `block_id` on a prediction row is the block
of the split being *evaluated*, not the block the model was *trained* on — for
`transfer_ben_to_hin` it reads `H`, not `B`. The source block is recoverable from
`run_id` and from the condition name, so nothing is lost, but a source-selection
table keyed naively on `block_id` would group by the wrong axis. T-408 must take the
source block from the condition, the way `results_tables._source_id_for` already
does.

---

## Working agreement

- **Baseline-first is not advisory here.** The GPU budget does not cover the full grid
  comfortably. One complete row beats six partial cells.
- **Report which cells did not run.** An incomplete grid stated plainly is a scoping
  decision; an incomplete grid presented as complete is a defect.
- **Report asymmetry honestly.** If `Hi→Bn` and `Bn→Hi` differ substantially, that is a
  finding, not an error to be explained away.
- Prefer boring code. This must be reproducible cold in October.
