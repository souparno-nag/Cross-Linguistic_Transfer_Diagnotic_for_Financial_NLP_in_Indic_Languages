# Cross-Linguistic Transfer Diagnostic for Financial NLP in Indic Languages

This project measures how much a financial-NLP model trained on one Indic language loses
when it is applied to another, and separates that loss into two causes that normally look
identical from the outside: the model genuinely failing to transfer across languages, and
the machine translation used to create test data in the other languages damaging the text
on the way. Financial text was chosen because it is unusually easy to check for damage —
a translated sentence either kept its numbers, currency symbols and percentages intact or
it did not, which gives an objective check that plain sentence quality does not.

The project has four completed phases:

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

A fifth phase (diagnosing *why* the model fails on specific items — e.g. does it lose
numerals, named entities, or long sentences disproportionately) is in progress
separately; nothing in this repository claims to explain a cause yet, only to measure
the size of the effect.

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
CLAUDE*.md                           Phase-scoped specifications (corpus / training / transfer / diagnostics / encoders)
docs/                                Detailed per-phase documentation (this repo's write-up)
src/                                 All importable pipeline logic, flat per phase
scripts/                             Thin CLI entry points, one per task (t1xx = Phase 1, t2xx = Phase 2, t3xx = Phase 3, t5xx = encoder comparison)
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

**So far this only covers transfer *out of* Hindi**, because only the Hindi classifier has
been trained (Phase 2 deliberately deferred Bengali and Telugu training to save GPU time).
42 of the 63 planned (condition × seed) evaluation runs are consequently still blocked,
not silently skipped — they are recorded as blocked in the results tables so the gap in
coverage is visible rather than hidden.

See `docs/phase3.md` for the full per-direction numbers and `reports/transfer_results.md`
for the generated tables.

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
- **Bengali and Telugu classifiers have not been trained yet.** All Phase 3 transfer
  numbers currently describe transfer *out of* Hindi only.
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

## Documentation

- `docs/phase1.md` — corpus construction, task by task (T-101–T-114)
- `docs/phase2.md` — training pipeline and Hindi baselines (T-201–T-208)
- `docs/phase3.md` — zero-shot transfer evaluation (T-301–T-308)
- `docs/phase5.md` — encoder comparison and the capacity-dilution question (T-500–T-505)
- `data/v1.0/DATASHEET.md` — the released corpus's datasheet (composition, per-direction
  translation quality, licensing), auto-generated from the frozen artifacts
- `reports/encoder_comparison.md` — every encoder on identical conditions, with gaps
  and confidence intervals
- `reports/capacity_dilution.md` — does the Indic-specialised encoder win, and where?
- `CLAUDE.md`, `CLAUDE2.md`, `CLAUDE3.md`, `CLAUDE4.md`, `CLAUDE5.md` — the original
  phase specifications this work was built against, including the hard rules and
  working agreements each phase follows

## License

The corpus is a derivative of IndicFinNLP and is distributed under **CC BY-NC-SA 4.0**
(attribution, non-commercial, share-alike) — see `data/v1.0/DATASHEET.md` and
`data/base_paper/license.txt` for full terms. Model licenses (IndicTrans2, LaBSE,
IndicBERT-v2) are separate and are documented where each model is introduced.
