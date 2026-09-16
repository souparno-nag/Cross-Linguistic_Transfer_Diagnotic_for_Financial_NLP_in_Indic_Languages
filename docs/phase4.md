# Phase 4 — Bengali and Telugu Source Training

Phases 2 and 3 trained a classifier on Hindi and measured what it lost when applied to
Bengali, Telugu and Malayalam. That answers "how well does a Hindi model travel", but it
cannot answer the question underneath it: **is Hindi a good language to have started
from?** With one source language there is nothing to compare against.

Phase 4 trains the same classifiers on Bengali and on Telugu, runs the identical
evaluation, and fills in the rest of the grid. That turns a single measurement into a
comparison, and it is the comparison that produces this phase's findings.

**The headline: Hindi is the worst of the three source languages**, on both encoders.
It is also the highest-resourced of the three and the one this project originally chose.

---

## What was run

Every cell of the matrix, on both encoders that fit on the card:

| | IndicBERT-v2 | mBERT | XLM-R |
|---|---|---|---|
| **Hindi source** | Phase 2/3 | Phase 5 | not runnable |
| **Bengali source** | Phase 4 | Phase 4 | not runnable |
| **Telugu source** | Phase 4 | Phase 4 | not runnable |

That is 63 of 63 (condition × seed) pairs on each task, for each encoder — 126 results
per task. Before this phase, 21 of 63 ran and 42 were reported as blocked.

XLM-R is excluded for the reason Phase 5 established: fine-tuning it needs 4.14 GiB of
weights, gradients and optimiser state before a single sentence is loaded, against
3.68 GiB of card. That is fixed cost, so no smaller batch rescues it. Nothing about
adding a source language changes that arithmetic, so Phase 4 inherits the exclusion
rather than re-deciding it.

### The pipeline really was language-agnostic

The phase brief predicted this should be "configuration, not code", and that held.
Adding a source language took eight YAML files and no changes to the training or
evaluation path: the config object works out which block a language belongs to,
checkpoint names are built generically, and the prediction, failure-set and
results-table code all take the block from the run or the condition rather than
assuming Hindi.

One guard was missing and had to be added. Phase 5 has a check that stops two
*encoders* drifting apart — it compares each config against the IndicBERT config **for
the same language**. That check has nothing to say about a Bengali IndicBERT config,
because that file *is* the IndicBERT config for its language; it compares it against
nothing and passes. A Bengali baseline could therefore have quietly carried a different
learning rate from the Hindi one it was about to be compared against, and the
source-comparison table would have reported a training-recipe difference as a
language difference. The new check compares each config against the **same-encoder
Hindi** config and allows only the language to move.

It is deliberately stricter than the encoder version. Phase 5 permits mBERT to split a
batch in half because a bigger model genuinely does not fit; changing the source
language changes neither the model nor the sentence length, so nothing about a block
can force that, and a config needing it would be tuning rather than accommodation.

---

## Do the three languages start from the same place?

This matters before any comparison. If one source model were simply better at its own
language, its transfer numbers would be flattered and the comparison would be about
model quality rather than transfer.

They do not differ meaningfully. In-language macro-F1 on the sustainability task:

| | Hindi | Bengali | Telugu |
|---|---|---|---|
| IndicBERT-v2 | 0.8226 ± 0.0044 | 0.8201 ± 0.0279 | 0.8191 ± 0.0148 |
| mBERT | 0.8742 ± 0.0170 | 0.8584 ± 0.0138 | 0.8602 ± 0.0094 |

Within a quarter of a point across all three languages, for each encoder. Whatever the
transfer differences turn out to be, they are not explained by one model being better
at home than another.

---

## The findings

Three results appear on both encoders. Two appear on one and not the other, and are
reported as properties of the encoder rather than of the languages — a distinction this
project has had to make before, and one that is easy to get wrong when only half the
grid has been run.

### 1. Hindi is the worst source language

Mean transfer gap across its three targets — lower means the model loses less when
moved to another language:

| encoder | Bengali | Telugu | Hindi |
|---|---|---|---|
| IndicBERT-v2 | **0.3448** | 0.3671 | 0.4291 |
| mBERT | **0.1534** | 0.1742 | 0.1956 |

Same ordering on both, despite the two encoders differing enormously in size and
architecture. Bengali is the best source to train on if the goal is a model that works
in other Indic languages; Hindi is the worst.

This is the result the original single-source design could not have produced. It is
also mildly uncomfortable: Hindi was chosen because it has the most data and the best
tooling, and on this evidence that is not the same thing as being the best starting
point for transfer.

### 2. The cost is in the target language, not the language pairing

The intuitive story is that transfer should be easy between related languages and hard
between distant ones. That is not what the numbers show. Pooling across every source
language, and asking only what the *target* was:

| encoder | into Dravidian | into Indo-Aryan |
|---|---|---|
| IndicBERT-v2 | 0.4055 | 0.3488 |
| mBERT | 0.1935 | 0.1505 |

Crossing *into* a Dravidian language (Telugu, Malayalam) costs more than crossing into
an Indo-Aryan one (Hindi, Bengali), regardless of where the model started. On
IndicBERT, Dravidian→Dravidian — same family, supposedly the easy case — is the single
worst quadrant of the four.

**Phase 1 found the same shape in the machine translation, by a completely different
method.** Ranking the nine translation directions by how much meaning they lost, it
concluded that "the cost is in the direction of travel, not the pairing". That was a
sentence-similarity measurement on translated text; this is a classifier's accuracy.
They share no code and no methodology, and they agree.

### 3. Transfer is not symmetric

Hindi→Bengali and Bengali→Hindi are the same two languages, the same content, and
models that are equally good at home. They are not equally hard:

| encoder | Hindi→Bengali | Bengali→Hindi | difference |
|---|---|---|---|
| IndicBERT-v2 | 0.4477 | 0.2631 | 0.1846 |
| mBERT | 0.1415 | 0.1052 | 0.0362 |

Both encoders make the same direction the harder one, and in both cases the difference
is larger than the run-to-run variation from changing the random seed. The
Bengali↔Telugu pair also looks asymmetric on one encoder, but the two disagree about
which direction is harder and mBERT's difference is inside its own noise, so that one
is not reported as a finding.

### 4 and 5. Two results that depend on the encoder

**Whether typological proximity predicts the best source.** For each target language
with both a same-family and a cross-family source available, does the same-family one
win? IndicBERT says yes in 1 case of 3; mBERT says yes in 3 of 3. The clearest
disagreement is Malayalam, which is Dravidian: on mBERT its best source is Telugu, the
fellow Dravidian language, exactly as proximity predicts; on IndicBERT, Telugu is its
*worst* source and Bengali wins.

Since the two encoders contradict each other on identical data, this cannot be reported
as a fact about the languages. It is recorded as unresolved.

---

## Task 3 shows none of this

Everything above is from the sustainability task. The ESG-theme task supports none of
it: every transfer gap clusters between 0.47 and 0.53 regardless of source, target or
encoder, and its target-family effect is flat.

That is the expected shape for a task sitting at its data ceiling. It has 532 rows
across 10 classes, and Phase 2's validation gate already established that the published
baseline for it is 0.05 macro-F1 — the constraint is the data, not the model. When
everything scores badly, everything scores about equally badly, and there is no signal
left for a comparison to find.

---

## A model that never trained looks like a model that transfers perfectly

This was the phase's most instructive mistake, and the reason a new check exists.

Three of the nine IndicBERT task-3 runs finished, logged a result, and appeared in the
baseline table without ever having learned anything — their loss barely moved from where
it started and they went on predicting a single class. On the *test* set that is
invisible: a healthy run on this task scores about 0.15, and so does a run that never
started, so the two are indistinguishable from the outside.

They are not remotely alike on the **training** set. A completed run on this task
memorises it and reaches 0.99; a failed one sits near the floor. So the check asks one
narrow question — did the model fit the data it was shown — and it looks at the training
fold, which is the only place the answer is legible.

The reason this matters beyond tidiness is what a broken model does to a transfer
number. **The gap is the model's score at home minus its score abroad.** A model that
cannot do the task scores equally badly in both places, so its gap is near zero — and a
near-zero gap reads as *excellent transfer*. Telugu's task-3 gaps initially read
0.16–0.17, far the best in the table, entirely because two of its three seeds had never
trained. With those excluded they read 0.4705, in line with everything else.

The same artefact had already reversed Bengali's task-3 comparison earlier in the phase.

So the baseline report now shows the converged-seed figure **alongside** the all-seeds
mean rather than instead of it — a run that failed to train is still a run that
happened, and the instability is part of the result — and the source-comparison table
excludes those seeds and names them.

### The threshold is a judgement, and it is stated

The check fires when a run's best training macro-F1 stays below 0.5. That number is a
judgement, so it is written down rather than buried, and a test asserts it is not a
knife edge: every run in the project sits at 0.15 or below, or at 0.52 or above, so
anything from 0.2 to 0.5 classifies them identically.

**That test currently fails**, deliberately. One Telugu run landed at 0.4692, right in
the middle of the band. Its instruction is explicit — do not move the threshold to make
it pass — so the run was read by hand instead, and it turns out to be a third kind of
outcome: not a failure to train, but a run cut off by early stopping while it was still
improving. Its loss had fallen 46% and its training score was still climbing when the
patience counter ran out. The genuine failures never got past 9%.

An earlier version of that test could not have caught this. It compared against a
hardcoded list of the scores observed the day it was written, so it stayed green when
new data arrived. It now reads the checkpoints.

### It is the encoder, not the schedule

The obvious response to three failed runs is that the training schedule is too short.
It is not. Every single mBERT run on the same task, the same data and the same schedule
reached a perfect training score:

| encoder | task-3 runs that fit their training data |
|---|---|
| IndicBERT-v2 | 6 of 9 |
| mBERT | **9 of 9** |

mBERT does not merely succeed more often — it reaches exactly 1.0000 in all nine runs,
with no variation, while IndicBERT ranges from 0.13 to 0.997 even among its successes.

IndicBERT-v2 is built on ALBERT, which reuses one layer's weights across all twelve, so
the part of it that actually does the computation is 7.8M parameters against mBERT's
85.6M. On roughly 370 training examples spread over 10 classes, a model that small
sometimes never escapes its starting point.

So the protocol was left alone. Raising the patience setting would have meant changing
it for every language and both encoders and re-running every task-3 baseline, to paper
over one encoder's fragility on a task where the other encoder needs no help. The
instability is reported as a property of IndicBERT-v2.

---

## Things that went wrong in the tooling

**A dropped network connection killed a 42-pair evaluation sweep three conditions in.**
Two separate faults, and the network blip only exposed them. First, the installed
version of `transformers` makes a live request to the model hub inside *every* tokenizer
load — to check whether the repository is a Mistral model — even when every file is
already cached locally; the tokenizer was uncached and loaded once per condition arm, so
a sweep made about seventy needless network calls. Second, the sweep had no error
handling, so any failure anywhere aborted everything, against the project's standing
rule to report a failure and continue. The tokenizer is now loaded once per encoder, and
a failed pair is reported and skipped.

**Running offline prints an alarming warning that is false.** With `HF_HUB_OFFLINE=1`
set, `transformers` takes a different branch and warns "This will lead to incorrect
tokenization". It does not. The warning branch only sets a bookkeeping flag; the branch
that actually changes the tokenizer needs an option nothing here passes. Verified twice
rather than argued: tokenizing 300 real Hindi sentences gives byte-identical token ids
either way, and re-running a whole condition under the flag reproduces all 3294
predictions with a maximum probability difference of zero.

**The comparison script spent longer loading checkpoints than bootstrapping.** It read
convergence by loading all 36 saved checkpoints — model weights and optimiser state
included — to look at one small field. It now reads the baseline report instead, which
takes two milliseconds and has the side benefit that the two documents cannot disagree
about which runs trained.

---

## Limitations

- **Neither validation gate has been scored.** The phase requires the Bengali and Telugu
  baselines to land within 2 points of IndicFinNLP's published monolingual numbers, and
  those published numbers are not in this repository — only the Hindi ones are. The runs
  are complete and healthy; the comparison against the paper has not been made.
- **Two encoders, not three.** XLM-R cannot be fine-tuned on this hardware. It and mBERT
  have identically-sized transformer bodies and differ mainly in vocabulary and
  pretraining breadth, which made that pair the one controlled test of whether breadth
  or size explains mBERT's advantage. That question stays open.
- **Task 3 supports no comparison**, for the data-ceiling reason above.
- **Three IndicBERT task-3 runs never trained**, and one more was cut off mid-climb.
  They are reported, not hidden, but they mean IndicBERT's task-3 numbers rest on fewer
  working runs than the seed count suggests.
- **Malayalam is never a source**, because no native Malayalam text exists upstream. So
  the Dravidian→Dravidian quadrant has exactly one cell, and "does a same-family source
  help" cannot be asked at all for Telugu as a target.
- **No causal explanation.** This phase measures which source transfers best. It does
  not say why, and deliberately makes no attempt — that is the diagnostic phase's work.

---

## Outputs

| Artefact | Path |
|---|---|
| Run log | `experiments.csv` |
| Baselines, with convergence | `reports/baselines.{md,parquet}` |
| Transfer results, all 9 cells | `reports/transfer_results.md` |
| Source comparison | `reports/source_comparison.{md,parquet}` |
| Per-instance predictions | `data/predictions/task_{n}/[{encoder}/]{condition}.parquet` |
| Failure sets | `data/failures/task_{n}/[{encoder}/]{condition}.parquet` |

## Reproducing

```bash
export HF_HUB_OFFLINE=1          # avoids the hub round-trip in every tokenizer load

python -m scripts.t401_smoke --device cuda            # the pipeline is language-agnostic
python -m scripts.t206_baseline --device cuda         # all 12 configs x 3 seeds

for t in 2 3; do for L in hin ben tel; do for s in 0 1 2; do
  python -m scripts.t302_predictions --task $t --run-id task${t}_${L}_indicbert_seed${s} --device cuda
done; done; done
python -m scripts.t305_run_conditions --task 2 --device cuda
python -m scripts.t305_run_conditions --task 3 --device cuda
# and again with --encoder mbert-base, and t302 with _mbert_ run-ids

python -m scripts.t307_failures --task 2
python -m scripts.t308_results
python -m scripts.t408_source_comparison
```

Training the full grid is about four GPU-hours on a 4 GB card — an afternoon, not the
three weeks the phase plan originally budgeted. The estimate was corrected against
measured epoch times rather than assumed.
