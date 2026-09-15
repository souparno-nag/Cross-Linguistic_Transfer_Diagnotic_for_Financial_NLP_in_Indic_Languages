# Phase 5 — Encoder Extension

Phases 2 and 3 fine-tuned one encoder, IndicBERT-v2, and measured how much it loses
when applied to a language it was not trained on. Phase 5 asks whether that choice of
encoder was the right one, by running the same protocol on other encoders and
comparing.

The motivating idea comes from Conneau et al.: pretraining a model on many languages
buys it transfer between them, but spends a fixed amount of model capacity across more
of them. Past some point the spreading-thin wins. IndicBERT-v2 is trained on roughly a
dozen Indic languages; mBERT on 104 and XLM-R on 100. If specialisation matters, the
Indic model should win on Indic tasks.

**It does not.** That is the phase's main result, and most of this document is about
how much of it we can actually explain.

---

## What was run, and what was not

| encoder | parameters | status |
|---|---|---|
| IndicBERT-v2 | 34M | the Phase 2/3 reference |
| mBERT base | 179M | trained and evaluated (T-502) |
| XLM-R base | 279M | **could not be run** — see below |

XLM-R does not fit on this project's GPU. The reason is worth spelling out because it
is not the obvious one and it caught us out once already.

When you fine-tune a model, the GPU holds more than the model. It holds the weights,
a gradient for every weight, and — because the optimiser is AdamW — two further
running averages per weight. Those are all full-precision numbers even when the
arithmetic is done in half precision, so the bill is **16 bytes per parameter**, and
none of it depends on how many sentences you put through at a time:

| encoder | parameters | fixed cost | headroom on a 3.68 GiB card |
|---|---|---|---|
| IndicBERT-v2 | 34M | 0.50 GiB | +3.18 GiB |
| mBERT | 179M | 2.65 GiB | +1.03 GiB |
| **XLM-R** | 279M | **4.14 GiB** | **−0.46 GiB** |

XLM-R is over budget before a single sentence is loaded, so no reduction in batch size
or sequence length can rescue it. Confirmed on the card: the run dies trying to
allocate 734 MiB, which is exactly XLM-R's word-embedding table in full precision.

Its configs are kept in `configs/train/deferred/`, protocol-identical to their
IndicBERT siblings and still covered by tests, so a bigger card can run them later
without anyone first having to work out whether they drifted.

### The measurement that was wrong

Phase 2's VRAM budget table said XLM-R fitted comfortably. It was wrong, and the way
it was wrong is instructive.

The probe ran one real training step and recorded the peak memory. But it drove that
step through PyTorch's gradient scaler, which **skips** the optimiser update when the
first batch's gradients overflow — and on the probe's synthetic input they reliably
did. AdamW allocates its two running averages lazily, inside the update it never
performed. So the table was measuring a step that had not fully happened, and
under-reported every encoder by 8 bytes per parameter.

The tell was visible in the committed numbers all along: XLM-R's recorded peak of
2.76 GiB was *below* its own weights-plus-gradients-plus-optimiser floor of 4.14 GiB,
which is arithmetically impossible for a completed step. Three of nine rows had that
property. The probe now forces the allocation, and every row carries the floor beside
the measured peak so the same mistake announces itself.

---

## T-500 — was the pipeline really encoder-agnostic?

The brief's premise was that adding an encoder should cost a config entry and nothing
else. The smoke test exists to check that before spending GPU hours on the assumption.

It was not true. Five things were hardcoded, and the two worst were found only after
the runs had started:

1. The evaluation sweep built the name of the checkpoint it needed by pasting the
   string `indicbert` into it, so it would have reported every condition as "blocked"
   for any other encoder and evaluated nothing.
2. The per-instance prediction log had no column saying which encoder produced it.
3. The prediction-log and failure-set paths had no encoder level, so a second
   encoder's results would land in the first's files.
4. The script that writes the **in-language ceiling** — the reference every transfer
   gap is measured against — ignored the encoder entirely. Running it for mBERT would
   have overwritten IndicBERT's ceiling and silently changed the meaning of every
   transfer number already computed and committed.
5. The results tables read source and target predictions from the default path, so a
   cross-encoder table would have paired one encoder's source predictions with
   another's target predictions and called the difference a transfer gap.

Points 4 and 5 are the ones worth remembering. Neither would have thrown an error.
Both would have produced plausible-looking numbers that were wrong.

The fix throughout was to make the encoder an explicit argument, with one deliberate
asymmetry: **IndicBERT keeps the original, un-namespaced paths.** Its Phase 3 artefacts
were committed before Phase 5, and the Phase 6 diagnostic pipeline reads them from
there with a non-recursive directory scan — so an encoder subdirectory is invisible to
Phase 6, and their in-progress work is untouched. Every other encoder gets its own
level.

---

## T-502 — mBERT, and one forced compromise

mBERT trained on both classification tasks at three seeds each. One config did not fit
as written: task 2 uses longer text, and mBERT ran out of room at 16 sentences per
batch.

The rule in the brief is that hyperparameters must not be tuned per encoder beyond
what memory forces, and that anything forced must be applied consistently and stated.
The fix chosen keeps the training identical in every way that affects learning: feed
the model **8 sentences at a time and update the weights after every two batches**.
Because the loss is an average, accumulating two half-batches produces the same
gradient as one full batch. The learning rate, schedule, number of updates and
effective batch size are all unchanged; only the number of sentences resident on the
card at any instant differs.

That was applied to mBERT alone. Matching it on IndicBERT would have meant retraining
Phase 2's baselines, re-running Phase 3's task-2 results, and re-validating the gate
that Phase 3 depends on — to erase a difference with no expected effect on the
outcome. The difference is stated instead: it appears in the config, in the baselines
report's `effective batch` column, and in the comparison report's header, and a test
permits `batch_size` and `grad_accum` to differ **only** when their product does not.

Getting there also produced a small methodological lesson. The first time task 2
OOM'd, it did so at the end of a long run that had already loaded and freed several
other models, and the error mentioned 562 MiB "reserved but unallocated" — the
signature of a fragmented allocator rather than a full card. That was a plausible
enough explanation to be worth testing, so the run was repeated in a fresh process
doing nothing else. It failed identically, which settled it. Had it passed, changing
the config would have been tuning that memory did not force.

---

## The results

### Both encoders are scored on identical rows

Worth establishing before any comparison: the train/test partition is seeded and does
not depend on the encoder, so for a given task and seed **both encoders saw exactly the
same test items** — the same 224 on task 2, the same 80 on task 3. This was verified,
not assumed.

### In-language, and across languages

| | IndicBERT-v2 | mBERT |
|---|---|---|
| task 2, in-language | 0.8226 ± 0.0044 | **0.8742 ± 0.0170** |
| task 3, in-language | 0.1526 ± 0.0249 | **0.3354 ± 0.0417** |
| task 2, cross-lingual (mean of 6 cells) | 0.519 | **0.787** |
| task 3, cross-lingual (mean of 6 cells) | 0.087 | **0.294** |

mBERT wins all twelve comparable transfer cells, by 0.15 to 0.37 macro-F1, and every
one of those differences is larger than the two encoders' seed-to-seed variation.

### The transfer gaps disagree between tasks

The *gap* is how much a model loses going from its own language to another, measured
on the same items:

| | IndicBERT gap | mBERT gap |
|---|---|---|
| task 2 | 0.38 – 0.47 | **0.12 – 0.24** |
| task 3 | 0.50 – 0.51 | 0.45 – 0.56 |

On task 2 mBERT starts higher *and* loses much less. On task 3 the two lose almost
exactly the same amount, and on Hindi→Malayalam mBERT's loss is the larger of the two.
"The better encoder transfers better" therefore holds on one task and not the other,
and should not be written up as a property of the encoders. This echoes Phase 1's
T-111, where two of three comparable questions also flipped between tasks.

### Translationese

Reading Hindi that was machine-translated from another language, rather than written
by a person, costs:

| | IndicBERT-v2 | mBERT |
|---|---|---|
| task 2 | −5.7 to −6.9 points | −2.4 to −3.3 |
| task 3 | −26.2 to −26.7 points | −18.0 to −18.9 |

Real for both encoders, consistently smaller for mBERT.

---

## T-505 — so is it capacity dilution?

The honest answer is that we cannot tell, and the reason is specific.

**The confound is bigger than the headline parameter counts suggest.** IndicBERT-v2 is
built on ALBERT, which reuses one layer's weights for all twelve layers. So although
its total is about 5× smaller than mBERT's, the part that actually does the
computation — the transformer body, excluding the vocabulary table — is **7.8M against
85.6M, or 11× smaller.** Most of IndicBERT's parameter count is a 200,000-entry
embedding table.

**The specialisation advantage is real, and it did not help.** Measured on the same
sentences in four languages, IndicBERT's tokenizer needs **14–24% fewer tokens** than
mBERT's, and its advantage grows the further the language sits from the centre of the
pretraining mix — 1.16× for Hindi, 1.32× for Malayalam. Neither tokenizer produces a
meaningful number of unknown tokens, so this is about how finely each one cuts, not
about coverage. The Indic model genuinely represents Indic text more compactly, and
still loses every task measure.

**The deficit is concentrated in transfer.** On task 2, IndicBERT is only 0.052 behind
in its own language but 0.268 behind across languages — a **5.2× amplification.** A
model that were simply too small would be behind by a similar margin everywhere. This
one is nearly competitive at home and collapses abroad; its cross-lingual average of
0.519 on a *binary* task is close enough to chance to be better described as not
transferring at all. (Task 3 shows no such amplification, but it should not be read as
contradicting this: IndicBERT only reaches 0.15 in-language there, so there is little
competence to lose.)

The tempting conclusion is that the Indic model has weaker cross-lingual *alignment* —
that its representations of Hindi and Bengali sit further apart. The evidence is
consistent with that and does not establish it, because the 11× capacity difference
predicts something similar. It is recorded as a hypothesis, not a finding.

### The experiment that would have settled it

There is an awkward coincidence in the parameter table: **mBERT and XLM-R have
identical transformer bodies** — 85.6M parameters, same width, same depth. What differs
between them is the vocabulary (119,547 against 250,002) and the number of pretraining
languages.

That pair is the controlled experiment capacity dilution calls for: hold capacity
fixed, vary breadth. It is exactly the arm that did not fit in memory. So XLM-R's
absence is not the loss of a third data point — it is the loss of the comparison that
would have answered the question. Phase 5 records the question as open rather than
answering it with the two encoders that happened to run.

---

## T-503 — why adapters were not used

Adapters (freezing the pretrained model and training small inserted layers) were the
brief's planned fallback for exactly this situation, and they **would** have worked:
freezing XLM-R removes the gradients and optimiser state for all 278M parameters and
brings it comfortably within budget.

They were rejected anyway. An adapter-trained XLM-R compared against two fully
fine-tuned encoders would differ in *how it was trained* as well as in what it was
pretrained on, so a poor score could mean capacity dilution or could mean adapters
simply have less room to adapt — and nothing in the results would separate those. That
is precisely the question T-505 exists to answer, made less answerable. Running all
three encoders with adapters would remove the confound, at the cost of reopening two
finished phases and a validation gate, for an extension the brief itself calls
deferrable.

There is also a practical risk: the MAD-X approach means AdapterHub's library, which
pins itself against particular `transformers` versions, and this project's pin is
load-bearing — it exists because the translation model used to build the frozen corpus
breaks on newer versions.

Full reasoning in `reports/t503_adapter_decision.md`.

---

## Limitations

1. **Two encoders, not three.** XLM-R is memory-blocked, and it was the controlled arm.
2. **The size confound is unresolved.** 11× body size and a different architecture
   family move together with the multilinguality difference.
3. **Hindi-source cells only.** Twelve of eighteen transfer cells per task are blocked
   for both encoders because Phase 2 deferred the Bengali and Telugu baselines, so
   every conclusion here concerns transfer *out of Hindi*.
4. **One protocol difference**, measured and stated: task 2's mBERT runs used a
   half-size micro-batch at an unchanged effective batch.
5. **Three seeds**, and task 3's seed spread is wide enough that its gap comparisons
   carry little weight.
6. **Task 1 is absent**, as in Phases 2 and 3: it is a numeral-span task with no
   classification label.

---

## Outputs

| Artefact | Path |
|---|---|
| Run log | `experiments.csv` |
| Baselines | `reports/baselines.{md,parquet}` |
| VRAM budget | `reports/vram_budget.{md,parquet}`, `reports/vram_epochs.parquet` |
| Predictions | `data/predictions/task_{n}/[{encoder}/]{condition}.parquet` |
| Failure sets | `data/failures/task_{n}/[{encoder}/]{condition}.parquet` |
| Cross-encoder comparison | `reports/encoder_comparison.{md,parquet}` |
| Capacity analysis | `reports/capacity_dilution.md` |
| Adapter decision | `reports/t503_adapter_decision.md` |

The `{encoder}` level is omitted for IndicBERT-v2, whose artefacts predate Phase 5 and
are what the Phase 6 pipeline currently reads.

## Reproducing

```bash
python -m scripts.t500_smoke --audit-only                      # instant, no model
python -m scripts.t205_vram --full-epoch                       # ~15 min, GPU
python -m scripts.t206_baseline --config configs/train/task3_hin_mbert.yaml --device cuda
python -m scripts.t206_baseline --config configs/train/task2_hin_mbert.yaml --device cuda
python -m scripts.t305_run_conditions --task 3 --encoder mbert-base --device cuda
python -m scripts.t305_run_conditions --task 2 --encoder mbert-base --device cuda
for s in 0 1 2; do python -m scripts.t302_predictions --task 3 --run-id task3_hin_mbert_seed$s --device cuda; done
for s in 0 1 2; do python -m scripts.t302_predictions --task 2 --run-id task2_hin_mbert_seed$s --device cuda; done
python -m scripts.t307_failures --task 3 --encoder mbert-base   # CPU
python -m scripts.t307_failures --task 2 --encoder mbert-base   # CPU
python -m scripts.t504_comparison                               # ~5 min, CPU
python -m scripts.t505_capacity                                 # CPU
```
