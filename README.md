# Cross-Linguistic Transfer Diagnostic for Financial NLP in Indic Languages

This project measures how much a financial-NLP model trained on one Indic language loses
when it is applied to another, and separates that loss into two causes that normally look
identical from the outside: the model genuinely failing to transfer across languages, and
the machine translation used to create test data in the other languages damaging the text
on the way. Financial text was chosen because it is unusually easy to check for damage —
a translated sentence either kept its numbers, currency symbols and percentages intact or
it did not, which gives an objective check that plain sentence quality does not.

The project has seven completed phases:

1. **Corpus construction** — build a parallel corpus across Hindi, Bengali, Telugu and
   Malayalam, where "parallel" means the same underlying content exists in every
   language, so a model can be evaluated on the *same thing* in four languages.
2. **Training** — fine-tune a classifier on Hindi and confirm it reproduces published
   numbers, so any later cross-lingual result is measured against a validated baseline
   rather than an unverified one.
3. **Zero-shot transfer evaluation** — run that frozen Hindi model on the other languages
   with no further training, measure how much accuracy is lost, and log every individual
   prediction so a later phase can investigate *why* specific items failed.
4. **Encoder comparison** — repeat the whole training and evaluation protocol on a
   second encoder, to check whether the choice of a specifically Indic model was the
   right one. It was not: a general-purpose multilingual model (mBERT) beat it on every
   measure. Why it beat it is not settled — see below.
5. **Source-language comparison** — train the same classifiers on Bengali and on Telugu
   too, so that "how well does a Hindi model travel" becomes "which language is the best
   one to start from". With one source language there is nothing to compare against;
   with three there is. The answer is that **Hindi is the worst of the three**, on both
   encoders — which matters because Hindi has the most data and was this project's
   original choice.

6. **Failure diagnosis** — give every wrong prediction exactly one cause label: the
   translation lost the meaning, a key word was split or had an ending attached, the
   model looked at different words, or — when no language cause fits — a pattern in the
   data or the model (the item fails in every language, only one training run got it
   wrong, the model predicts one class for everything). What fits nothing is labelled
   `unattributed` and reported, not hidden.
7. **Root-cause findings** — count those labels across every experiment and ask whether
   the mix of causes explains the size of the effects found in phases 3–5. Mostly it
   does not: see below.

## Why this exists

Most cross-lingual NLP evaluations translate a test set with machine translation and then
report a language's score as if it reflected the model's transfer ability. That
conflates two very different failures: the model not understanding the target language,
and the translator having quietly mangled a number, dropped an entity, or reworded a
sentence into something else. In financial text this matters more than usual, because a
model that can't tell "10 crore" from "10 million" is not making a subtle transfer
error — it is unusable. This project builds the corpus so those two failures can be told
apart, rather than assuming they don't happen.

## The three tasks

Everything runs over three financial-NLP tasks drawn from **IndicFinNLP** (Ghosh et al.,
LREC-COLING 2024), the only upstream dataset used:

| Task | What it is | Type | Native languages upstream |
|---|---|---|---|
| `task_1` | Financial sentences with a number marked by its character position | Span annotation, not classification | Hindi, Bengali, Telugu |
| `task_2` | Sentences about corporate sustainability | Binary: `sustainable` / `unsustainable` | Hindi, Bengali, Telugu |
| `task_3` | ESG (environmental/social/governance) news headlines | 10-class topic | Hindi, Bengali, Telugu |

IndicFinNLP does not ship Malayalam at all; every Malayalam row in this project exists
only because it was machine-translated in Phase 1. That is a deliberate, documented gap,
not an oversight — see the "Known limitations" section below.

**One finding shaped everything downstream and is worth understanding before anything
else here makes sense.** Tasks 2 and 3's Hindi, Bengali and Telugu versions turned out to
be the *same sentences*, translated by professional human translators and shuffled into
different row orders — not three independently written corpora. Task 1 is the only task
where the three languages hold genuinely different content. This was discovered by an
audit early in Phase 1 (see `docs/phase1.md`, T-102) and it changes what a "transfer"
result even means for tasks 2 and 3: without first figuring out which row in Hindi
corresponds to which row in Bengali, a model could be "evaluated" on a sentence it was
trained on, just written in a different language. Recovering that correspondence
(alignment) is why Phase 1 exists in the shape it does, and it also turned out to be a
gift: since the same sentence exists in a human translation and a machine translation,
the human version becomes a free quality ceiling for judging the machine one.

## Repository layout

```
CLAUDE*.md                           Phase-scoped specifications (corpus / training / transfer / diagnostics / encoders / source comparison)
docs/                                Detailed per-phase documentation (this repo's write-up)
src/                                 All importable pipeline logic, flat per phase
scripts/                             Thin CLI entry points, one per task (t1xx = corpus, t2xx = training, t3xx = transfer, t4xx = source comparison, t5xx = encoder comparison)
configs/                             Frozen configs: labels, translation/verification settings, eval conditions, training YAMLs
data/
  base_paper/     upstream IndicFinNLP release, committed as-is with checksums
  raw/            per-task, per-block corpus splits before freezing
  v1.0/           the frozen, immutable release corpus + DATASHEET.md
  verification/   LaBSE embeddings, similarity scores, alignment maps
  predictions/    per-instance model predictions (Phase 3), namespaced per encoder
  failures/       the subset of predictions that are genuine transfer failures (Phase 3)
reports/          generated audit/quality/results reports (Markdown + Parquet), one directory per task
checkpoints/      trained model weights (gitignored; regenerate with scripts/t206_baseline.py)
dashboard/        Streamlit results dashboard: transfer, causes, sentence explorer (read-only)
tests/            one test file per src/ module
```

Every corpus-facing module goes through `src/corpus_io.py`; nothing else is allowed to
read or write the corpus's Parquet files directly, which is what keeps the schema and
label rules enforced in one place instead of copy-pasted everywhere.

## The pipeline, end to end

```
IndicFinNLP (Hindi/Bengali/Telugu, human-written)
        │  T-102 audit: are the 3 languages independent, or the same content shuffled?
        ▼
Phase 1 — corpus construction
  ├─ T-102b  recover which row = which item across languages (tasks 2, 3 only)
  ├─ T-103   canonical IDs + label schema + Parquet IO layer
  ├─ T-104   IndicTrans2 translation pipeline, frozen decoding config
  ├─ T-105   check that numbers/currency/percentages survive translation, by value
  ├─ T-106   generate all 9 translation directions × 3 tasks (92,760 rows)
  ├─ T-107   structural integrity + corruption-mode detection
  ├─ T-108   LaBSE similarity score for every machine-translated pair
  ├─ T-109   per-direction drift report
  ├─ T-110   calibrate the similarity threshold τ per task, empirically
  ├─ T-111   rank the 9 translation directions by reliability
  ├─ T-112   freeze data/v1.0/ (immutable, checksummed)
  ├─ T-113   enumerate valid train/eval conditions (no accidental train=eval leakage)
  └─ T-114   generate DATASHEET.md from the artefacts above
        │
        ▼   data/v1.0/  (frozen corpus, read-only from here on)
Phase 2 — training pipeline
  ├─ T-201   environment / dependency pinning
  ├─ T-202   dataset loader (native splits only, task-aware)
  ├─ T-203   training loop (encoder → [CLS] → linear head)
  ├─ T-204   YAML config system + config hashing
  ├─ T-205   VRAM budgeting for a 4 GB GPU
  ├─ T-206   train IndicBERT-v2 on Hindi, 3 seeds, tasks 2 and 3
  ├─ T-207   validate against IndicFinNLP's published baseline numbers   ⚠ gate
  └─ T-208   checkpoint naming, retrieval, retention
        │
        ▼   checkpoints/  (frozen Hindi-trained model weights)
Phase 3 — zero-shot transfer evaluation
  ├─ T-301   load a frozen checkpoint, run inference only (assert no gradient step)
  ├─ T-302   per-instance prediction log (item_id, gold, predicted, probabilities)
  ├─ T-303   accuracy / macro-F1 / per-class F1 / confusion matrix
  ├─ T-304   bootstrap confidence intervals + transfer gap (source − target)
  ├─ T-305   run every valid evaluation condition × 3 seeds
  ├─ T-306   flag "mismatches": right in the source language, wrong in the target
  ├─ T-307   export the failed-instance set (input to a future diagnostic phase)
  └─ T-308   render the final results tables
        │
        ▼
Phase 4 — source-language comparison
  ├─ T-401   confirm a new source language is a config change, not a code change
  ├─ T-402   train on Bengali, 3 seeds                                        ⚠ gate
  ├─ T-403   train on Telugu, 3 seeds                                         ⚠ gate
  ├─ T-404   evaluate every Bengali-source condition
  ├─ T-405   evaluate every Telugu-source condition
  ├─ T-406   repeat the whole grid on the second encoder
  ├─ T-407   export the failed-instance sets for the diagnostic phase
  └─ T-408   which source transfers best, and does language family predict it?
        │
        ▼   data/failures/  (right in the source language, wrong in the target)
Phase 6 — failure diagnosis
  ├─ T-601   router: semantic gate first, then linguistic modules, one label each
  ├─ T-602   tokenizer fragmentation
  ├─ T-604   morphological masking (a grammatical ending hides the key word)
  ├─ T-605   saliency divergence via Integrated Gradients
  ├─ T-606   audit trail of every module that fired
  ├─ T-608   hand spot-check of the labels                                  98.5%
  └─ T-609   data and model-behaviour labels for what language does not explain
        │
        ▼   data/diagnostics/  (one cause label per failure)
Phase 7 — root-cause findings
  ├─ T-701   share of each label in every one of the 72 experiment cells
  ├─ T-702   by target language and direction (is "into Dravidian" different?)
  ├─ T-703   by source language (why is Hindi the worst start?)
  ├─ T-704   by encoder (what does mBERT do differently?)
  ├─ T-705   native vs machine-translated test text
  ├─ T-706   which differences hold on both encoders
  └─ T-707   generate reports/findings.md from the above
```

T-207 is a hard gate: Phase 3 is not allowed to start until the Hindi baseline
reproduces IndicFinNLP's published numbers within tolerance (or the gap is written up and
understood). An unvalidated baseline would make every transfer number downstream
meaningless, because there would be no way to tell "the model transfers badly" from "the
model was never trained properly in the first place."

## Results so far, in plain terms

**The Hindi baseline reproduces the published paper**, so the transfer numbers below rest
on a validated starting point rather than an unverified model (Phase 2, T-207).

**Zero-shot transfer loses roughly half the model's accuracy**, consistently, across both
classification tasks and every language it was tested on:

| | Task 2 (sustainable / unsustainable) | Task 3 (10-way ESG topic) |
|---|---|---|
| Hindi (in-language) macro-F1 | 0.95 | 0.59 |
| Best cross-language macro-F1 | 0.57 (Malayalam) | 0.09 |
| Typical accuracy lost moving to another language | ~0.39–0.45 | ~0.50–0.51 |

That gap is far larger than the run-to-run noise from changing the random seed, so it is
a real effect, not a measurement artifact (Phase 3, T-304/T-308).

**Whether the target text was written by a human or produced by machine translation
barely changes the size of that gap.** If translation damage were the main driver of the
score drop, evaluating on real human-written Bengali should score noticeably better than
evaluating on machine-translated Bengali. It doesn't — the two conditions land within a
few points of each other. That is evidence the drop is a genuine cross-language transfer
problem in the model, not mostly an artifact of this project's own translation pipeline
(Phase 3, T-308).

**Every planned evaluation now has a result.** All 63 (condition × seed) pairs run on
each task, for each of the two encoders that fit on the available GPU — 126 results per
task. Earlier versions of this README reported 42 of 63 as blocked, because only the
Hindi classifier existed; Bengali and Telugu have since been trained.

**Which language you train on matters, and the obvious choice is the wrong one.**
Training on Bengali produces a model that loses least when moved to other languages;
training on Hindi loses most. The three source models are equally good *in their own
language* (within a quarter of a point of each other), so this is about transfer, not
about one model being better. Both encoders agree on the ordering.

**The difficulty lives in the target language, not in the language pairing.** Moving a
model into a Dravidian language (Telugu, Malayalam) costs more than moving it into an
Indo-Aryan one (Hindi, Bengali), whatever language it started from — and "same family"
does not help: on one encoder, Dravidian→Dravidian is the worst case of all. Phase 1
reached the same conclusion about the machine translation itself, using a completely
different measurement, which makes it the most solid finding here.

**Transfer is not symmetric.** Hindi→Bengali loses noticeably more than Bengali→Hindi —
same two languages, same sentences, equally capable models — on both encoders and by
more than seed-to-seed noise.

Average accuracy lost when moving a model out of each source language (lower is a
better language to train on), on the sustainability task:

| trained on | IndicBERT-v2 | mBERT |
|---|---|---|
| **Bengali** | **0.34** | **0.15** |
| Telugu | 0.37 | 0.17 |
| Hindi | 0.43 | 0.20 |

See `docs/phase3.md` for the per-direction numbers out of Hindi, `docs/phase4.md` for
the source comparison, and `reports/{transfer_results,source_comparison}.md` for the
generated tables.

### What the failures are made of (phases 6 and 7)

A "failure" here is an item the model got right in its training language and wrong in
the target language. Every one got a cause label; `reports/findings.md` counts them.
Each share below is a share of *failures*: it says what the failures are made of, not
that a label makes failure more likely.

**Most failures are explained by where they happen, not by the sentence.** Across the
four task-and-encoder combinations, 9–20% of failures fit no cause at all, 9–12% get a
language-level cause, and 69–79% get a data or model-behaviour label: the item fails in
every target language, fails on only one kind of test text, fails on only some training
runs, or the model has collapsed to predicting one class. Those labels describe a
pattern; they do not name a mechanism.

**The extra cost of moving into Telugu or Malayalam does not come with extra language
causes.** At first sight it does: failures going into Telugu get the "grammatical ending"
label far more often than failures coming out of it. But the comparison has Hindi on one
side, and that label can never fire on Hindi, which writes those endings as separate
words. Comparing Bengali↔Telugu instead, with no Hindi involved, the difference no
longer passes the test: the language labels as a whole are within noise, and the
grammatical-ending label separates on one encoder only. The one difference that does hold
runs the other way: on the sustainability task, failures coming *out of* Telugu are more
often model-behaviour failures than those going in.

**Hindi fails more often, but no differently that the test can see.** Between Hindi- and
Bengali-source failures, no difference in the mix of causes passed the test on either
task. Hindi simply fails more (33% vs 28% of items on IndicBERT-v2, 17% vs 13% on
mBERT, sustainability task).

**mBERT and IndicBERT-v2 fail in different ways.** IndicBERT-v2 fails twice as often on
the sustainability task, and 27% of its failures come from collapsing to one class, which
mBERT never does. mBERT's failures are more often unstable across training runs, or
specific to one kind of test text.

**Our own translation is not the main cause.** On machine-translated test text, about
10–25% of failures are labelled as the translation's doing. Apart from the drift label,
whose native-vs-MT difference is an artefact of how drift is measured, no difference
between human-written and machine-translated test text passed the test.

All of this comes with the limits in `reports/findings.md` §7: two of the language labels
are no better than chance at predicting failure, and the hand spot-check covered one of
the four task-and-encoder combinations.

## Known limitations

- **No native Malayalam anywhere upstream.** Every Malayalan row in the corpus is
  machine-translated, so Malayalam results always mix "does the model transfer" with
  "did the translation survive," in a way Hindi/Bengali/Telugu results do not.
- **Tasks 2 and 3 are not independently-sourced across languages** — their native splits
  are human translations of each other, discovered during the Phase 1 audit. This is why
  an alignment step and an item-level train/eval partition are mandatory for those two
  tasks (see `docs/phase1.md`, T-102 and T-113); skipping either would let a model be
  "evaluated" on sentences it was trained on.
- **No native-speaker review of the translated corpus.** A project maintainer spot-checked
  20 sentences per translation direction (about 1–4% of each task's machine-translated
  rows, depending on the task); nobody on the project reads Bengali, Telugu or Malayalam,
  so nothing beyond that sample has been read by a human for meaning. The automated
  checks (Phase 1, T-105/T-107/T-108) confirm structure, numerals and script survive —
  not that every sentence means the right thing.
- **The similarity score used to catch bad translations rewards literal wording, not
  quality.** Machine translations score *higher* on it than human translations of the
  same sentences, in every case where both exist to compare — a human translator
  paraphrases, and the metric penalizes that. It is trustworthy for ranking translation
  directions against each other, not as an absolute quality bar.
- **Neither Bengali nor Telugu baseline has been checked against the published paper.**
  Phase 2 validated the Hindi classifier against IndicFinNLP's published number before
  any transfer claim was made. The equivalent Bengali and Telugu numbers are not in this
  repository, so that check has not been performed for them — the models train healthily
  and match Hindi in-language, but they have not been gated the way Hindi was.
- **Three of the nine IndicBERT runs on the 10-class task never actually trained**, and
  one more was stopped while it was still improving. They are detected, reported
  alongside the full average rather than hidden, and excluded from the source
  comparison — a model that never learned scores equally badly at home and abroad, so it
  produces a near-zero "transfer gap" that looks like excellent transfer. mBERT has no
  such failures on identical data and settings, so this is a property of the smaller
  encoder rather than of the task.
- **The 10-class ESG task supports no source comparison.** Every transfer gap on it
  lands between 0.47 and 0.53 regardless of source, target or encoder. With 532 rows
  across 10 classes it is at its data ceiling — the published baseline for it is 0.05 —
  and when everything scores badly there is no signal left to compare.
- **No diagnostic/causal analysis yet.** Phase 3 identifies *which* individual predictions
  flip from correct to wrong when moving to another language and exports that set; it
  does not yet say *why* (numeral loss, entity loss, sentence length, etc.). That is the
  next phase, in progress separately.
- **The encoder comparison has two models, not three.** XLM-R could not be fine-tuned
  on the available 4 GB GPU — its weights, gradients and optimiser state need 4.14 GB
  before a single sentence is loaded — so it is deferred rather than run. That matters
  more than losing one data point: XLM-R and mBERT have *identical-sized* transformer
  bodies and differ mainly in vocabulary and pretraining breadth, which made that pair
  the one controlled test of whether breadth or size explains the result. The question
  is therefore recorded as open, not answered.
- **Why mBERT wins is not established.** It is 179M parameters to IndicBERT-v2's 34M,
  and because IndicBERT reuses one layer's weights across all twelve, the gap in the
  part that does the computation is 11×, not 5×. Size and multilinguality move together
  here and this design cannot separate them.

## Environment

- **Python 3.11**, exactly — every entry point asserts this and fails loudly otherwise.
- **Linux or WSL2** — the IndicTrans2 toolkit is not supported on native Windows.
- One GPU is assumed available: development was done on a 4 GB laptop GPU (RTX 3050),
  and the code is written to fit and checkpoint within that budget (adaptive batching,
  `expandable_segments` allocator setting, resumable jobs).
- `transformers` is pinned below version 5 — IndicTrans2's released model code is
  written against the 4.x API and breaks on 5.x in ways that cannot be safely patched
  around. See `docs/phase1.md` §Environment for the full explanation.
- Set up: `pip install -r requirements.txt` inside a Python 3.11 virtualenv. Model
  downloads (IndicTrans2, IndicBERT-v2) are gated on Hugging Face and require accepting
  each model's license and authenticating (`hf auth login`) before first use.

Run everything as a module from the repository root, e.g. `python -m scripts.t206_baseline`
— this is required for the internal path-resolution logic to find the repo root correctly.

**Long-running jobs (translation generation, model training) are meant to be run by a
human, not launched unattended** — they take hours and depend on intermittent GPU
availability. The code is built to checkpoint and resume rather than to be babysat.

## Dashboard

```
streamlit run dashboard/app.py
```

Three pages, switched by task in the sidebar:

- **Transfer** — how much macro-F1 each model loses from its training language to each
  test language, as one heatmap per model.
- **Causes** — what each setup's failures are made of (Phase 7's label shares), on native
  or machine-translated test text, by cause group or by individual cause.
- **Explorer** — one failed sentence at a time: the training-language and test-language
  text, every training run's prediction on both models, the cause each failure got, and
  every check's result behind it.

It only reads artefacts that already exist (`reports/`, `data/v1.0/`, `data/predictions/`,
`data/diagnostics/`), loads no model and needs no GPU.

## Documentation

- `docs/understanding_the_project.md` — start here if the domain is new: background, design
  reasoning, all seven phases, code map, results and limitations, glossary
- `docs/finetuning_architecture.md` — beginner's guide to the classifier and training loop:
  model, data pipeline, one training step, settings, memory limits, how to read a run
- `docs/phase1.md` — corpus construction, task by task (T-101–T-114)
- `docs/phase2.md` — training pipeline and Hindi baselines (T-201–T-208)
- `docs/phase3.md` — zero-shot transfer evaluation (T-301–T-308)
- `docs/phase4.md` — Bengali and Telugu source training; which language is the best one
  to start from (T-401–T-408)
- `docs/phase5.md` — encoder comparison and the capacity-dilution question (T-500–T-505)
- `data/v1.0/DATASHEET.md` — the released corpus's datasheet (composition, per-direction
  translation quality, licensing), auto-generated from the frozen artifacts
- `reports/encoder_comparison.md` — every encoder on identical conditions, with gaps
  and confidence intervals
- `reports/capacity_dilution.md` — does the Indic-specialised encoder win, and where?
- `reports/source_comparison.md` — every source language against every target, on every
  encoder: which source transfers best, and does language family predict it?
- `reports/findings.md` — what the failure-cause labels add up to, with every claim
  pointing to its row in `reports/phase7/robustness.md`
- `reports/diagnostics_summary.md` — the cause-label distribution per experiment
- `CLAUDE.md`, `CLAUDE2.md` … `CLAUDE7.md` — the original
  phase specifications this work was built against, including the hard rules and
  working agreements each phase follows

## License

The corpus is a derivative of IndicFinNLP and is distributed under **CC BY-NC-SA 4.0**
(attribution, non-commercial, share-alike) — see `data/v1.0/DATASHEET.md` and
`data/base_paper/license.txt` for full terms. Model licenses (IndicTrans2, LaBSE,
IndicBERT-v2) are separate and are documented where each model is introduced.
