# Understanding this project from zero

A guide for someone who knows little about Indic languages, financial NLP, machine
translation or cross-lingual transfer, and wants to understand this repository deeply.

It does not replace the per-phase write-ups (`docs/phase1.md` … `docs/phase5.md`), the
phase specifications (`CLAUDE*.md`) or the generated reports. It sits above them. It
explains the vocabulary, the reasoning, the shape of the system and where each fact
lives, so that those documents become readable.

> **How accurate is this?** It was written from the specs, the README, the generated
> reports, the configs and a sample of the real Parquet files. Numbers are quoted from
> those artefacts and carry a pointer to their source. The specs are long-lived and were
> revised in place, so where a spec and a generated report disagree, the generated report
> wins. Nothing here was produced by re-running the pipeline.

---

## Contents

1. [The project in one page](#1-the-project-in-one-page)
2. [Background you need first](#2-background-you-need-first)
3. [The research question and why the design looks the way it does](#3-the-research-question-and-why-the-design-looks-the-way-it-does)
4. [The data: what is in the corpus](#4-the-data-what-is-in-the-corpus)
5. [The seven phases, one at a time](#5-the-seven-phases-one-at-a-time)
6. [How the artefacts flow through the repo](#6-how-the-artefacts-flow-through-the-repo)
7. [Code map](#7-code-map)
8. [Engineering rules and the reasons behind them](#8-engineering-rules-and-the-reasons-behind-them)
9. [What was found, and how far to trust it](#9-what-was-found-and-how-far-to-trust-it)
10. [Limitations and open questions](#10-limitations-and-open-questions)
11. [Running things](#11-running-things)
12. [Glossary](#12-glossary)
13. [Where to read next](#13-where-to-read-next)

---

## 1. The project in one page

**The question.** You train a model to classify financial text in Hindi. You then apply it,
with no further training, to the same content in Bengali, Telugu or Malayalam. It does much
worse. *Why?*

There are two very different possible reasons, and from the outside they look identical:

- **Transfer failure.** The model learned something Hindi-specific and genuinely does not
  carry over to the other language.
- **Translation damage.** To get "the same content in Malayalam" you had to machine-translate
  it, and the translator quietly dropped a number, garbled an entity or reworded the
  sentence. The model is being tested on something that is no longer the original content.

Most published cross-lingual evaluations never separate the two. This project builds the
machinery to separate them, and then to go further and give every individual failure a
cause.

**Why finance.** Financial text has checkable content. A sentence either kept its numbers,
currency symbols and percentages through translation or it did not. That gives an objective
damage test that "is the translation fluent?" does not.

**What was built, in order.**

| Phase | What | One-line purpose |
|---|---|---|
| 1 | Parallel corpus | Same content in 4 languages, machine-translated, screened for damage, frozen |
| 2 | Training | Fine-tune a classifier on Hindi; prove it reproduces a published number |
| 3 | Zero-shot transfer | Run that frozen model on other languages; measure the loss; log every prediction |
| 4 | More source languages | Train on Bengali and Telugu too; which language is the best start? |
| 5 | A second encoder | Repeat everything with mBERT; was choosing an Indic-specific model right? |
| 6 | Failure diagnosis | Give every wrong prediction exactly one cause label |
| 7 | Root-cause findings | Count the labels; what explains the size of the effects? |

**What was found, in three sentences.** Zero-shot transfer loses roughly half the model's
accuracy, and that loss barely changes whether the test text is human-written or
machine-translated, so it is mostly a real transfer problem and not mostly an artefact of
our translation. Hindi, the language with the most data, turned out to be the *worst*
language to start from. Most individual failures are explained by *where* they occur (the
item fails in every language; the model collapsed to one class) and not by anything about
the target sentence's language, and 9–20% fit no cause at all.

**Naming trap you will hit immediately.** The spec files are not numbered by phase in
order:

| File | Phase it specifies |
|---|---|
| `CLAUDE.md` | 1 — corpus |
| `CLAUDE2.md` | 2 — training |
| `CLAUDE3.md` | 3 — zero-shot transfer |
| `CLAUDE6.md` | **4** — Bengali/Telugu sources |
| `CLAUDE5.md` | 5 — encoder extension |
| `CLAUDE4.md` | **6** — diagnostics |
| `CLAUDE7.md` | 7 — findings |

Task IDs follow the phase's hundred: `T-1xx` is Phase 1, `T-2xx` Phase 2, `T-3xx` Phase 3,
`T-4xx` Phase 4, `T-5xx` Phase 5, `T-6xx` Phase 6, `T-7xx` Phase 7. Those IDs are the most
reliable way to navigate: they appear in script names (`scripts/t106_generate.py`), reports
and commit messages.

---

## 2. Background you need first

Skip what you already know. Everything later assumes it.

### 2.1 The four languages

| Language | Code | Script | Family | Native data upstream? |
|---|---|---|---|---|
| Hindi | `hin` | Devanagari | Indo-Aryan | yes |
| Bengali | `ben` | Bengali (Bangla) | Indo-Aryan | yes |
| Telugu | `tel` | Telugu | **Dravidian** | yes |
| Malayalam | `mal` | Malayalam | **Dravidian** | **no** |

- **Two families that are not related.** Indo-Aryan languages descend from Sanskrit and are
  related to one another much as French and Spanish are. Dravidian languages (Telugu,
  Malayalam, Tamil, Kannada) are a separate family with a different grammar and a heavily
  different vocabulary. This is the central typological axis of the project: the "four
  quadrants" in the spec are Indo-Aryan→Indo-Aryan, Indo-Aryan→Dravidian,
  Dravidian→Indo-Aryan, Dravidian→Dravidian.
- **Four different scripts.** Each language uses its own writing system. A Hindi sentence
  and its Malayalam translation share *no* characters. A model's tokenizer therefore sees
  entirely different pieces for each language.
- **Morphology matters for one of the diagnostic modules.** Hindi writes grammatical
  relations as separate words: *"of the company"* is roughly `कंपनी के` (two words).
  Bengali attaches endings to the noun. Telugu and Malayalam are **agglutinative**: they
  glue many suffixes onto a root, so one written word can mean *"in the company's emissions
  (accusative)"*. A keyword like "emissions" can therefore be present in a sentence while
  the exact string you would search for never appears, because the root is buried under
  endings. The project calls that **morphological masking**.
- **Malayalam is a target only.** IndicFinNLP ships Hindi, Bengali and Telugu. Every
  Malayalam row in this project is machine-translated. That is a permanent, documented
  limitation: any Malayalam result mixes "does the model transfer" with "did the
  translation survive".

### 2.2 Numbers, scripts and Unicode — the part that bites

Financial text is full of numbers, and Indic numbers are awkward in three ways.

1. **Scale words and grouping.** India uses *lakh* (1,00,000 = 10⁵) and *crore*
   (1,00,00,000 = 10⁷), written with different digit grouping (`12,34,567`). English-style
   text says *million* (10⁶). So `100 million` and `10 crore` are the **same quantity**
   (10⁸) with different digits. A checker that compares digit strings reports a loss that
   did not happen. The project's numeral checker compares **values**.
2. **Native digits.** Each script has its own digit characters (`०१२३` Devanagari, `০১২৩`
   Bengali, `౦౧౨౩` Telugu, `൦൧൨൩` Malayalam) beside ASCII `0123`. Hindi and Telugu source
   text mostly uses ASCII digits; Bengali source text mostly uses Bengali digits. Machine
   translation output is ASCII in every language. Ranges are in CLAUDE.md §9 and
   `src/unicode_ranges.py`.
3. **Commas.** In `22, 2019` the comma separates two numbers. In `25,000` it groups
   digits. Rule used: a comma groups only when exactly three digits follow.

**Unicode normalisation is a trap.** Some letters can be stored as one code point or as a
base letter plus a combining mark (the *nukta*, a dot that modifies a consonant: `ड़`, `য়`).
Normalising (NFC) changes the string's length. Task 1 annotates numbers by **character
offset into the raw string**, so normalising silently shifts 880 of 22,786 annotations to
point at the wrong place, while the text looks fine. Hence hard rule 9: never normalise
task 1 text without recomputing offsets.

**The danda.** `।` (U+0964) is the Indic full stop. It lives inside the Devanagari Unicode
block but is used by Bengali, Telugu and Malayalam too. A naive "does this text contain
Devanagari?" check flags essentially every Bengali sentence. All script tallies exclude it.

### 2.3 The upstream dataset: IndicFinNLP

IndicFinNLP (Ghosh et al., LREC-COLING 2024) is the only external dataset used. Three tasks,
each in Hindi, Bengali and Telugu:

| Task | Content | Label | Size (rows) |
|---|---|---|---|
| `task_1` | Financial sentences, a number marked by character position | none; it is **span annotation** | 22,786 across 3 languages |
| `task_2` | Corporate-sustainability sentences | binary: `sustainable` / `unsustainable` | 6,538 |
| `task_3` | ESG news headlines | 10 classes (climate change, human capital, pollution waste, …) | 1,596 (532 each) |

ESG = Environmental, Social, Governance.

The 10 task-3 classes are listed in `configs/labels.json`; their order **is** the label ID
and is frozen.

### 2.4 Machine-learning vocabulary

You need these terms and nothing deeper.

- **Encoder / BERT-style model.** A neural network that turns a sentence into vectors. Here
  it is pretrained on large text, then **fine-tuned**: a small classification layer is added
  on top and all weights are trained on the labelled task. The project's classifier is
  deliberately minimal: encoder → vector at the `[CLS]` position → one linear layer →
  cross-entropy. Nothing else.
- **The two encoders.**
  - **IndicBERT-v2** (`ai4bharat/indic-bert`, 34M parameters): specialised for Indian
    languages. It is ALBERT-based, meaning one transformer layer's weights are reused for
    all twelve layers, so its computing "body" is only 7.8M parameters.
  - **mBERT** (`bert-base-multilingual-cased`, 179M parameters): a general 100-language
    model, body 85.6M. Roughly 11× the body of IndicBERT-v2.
  - XLM-R was planned and **could not be fine-tuned** on the 4 GB GPU (its fixed
    weight+optimizer cost is 4.14 GiB on a 3.68 GiB card).
- **Macro-F1.** The F1 score averaged equally over classes. It is the primary metric because
  accuracy hides failure under class imbalance (a model predicting the majority class gets
  a decent accuracy and a terrible macro-F1).
- **Seeds.** Every training run is repeated with 3 random seeds and reported as mean ± std.
  A gap smaller than seed-to-seed spread is not a finding.
- **Zero-shot transfer.** Train in language A, test in language B with **zero** further
  weight updates. The code asserts the model is in eval mode and no gradient graph is built.
- **Transfer gap.** `source in-language score − target score`, both measured on the same
  items. The source in-language number is the model's own ceiling.
- **Bootstrap confidence interval.** Resample the test items with replacement 1000 times,
  recompute the metric each time, take the 2.5th–97.5th percentile. Seeded, so reproducible.
- **Overfitting vs "never trained".** Task 3 has ~370 training rows for 10 classes. Models
  memorise it (train F1 ≈ 1.0, test ≈ 0.15). Some IndicBERT-v2 runs never learned at all
  (train F1 ≈ 0.15). These are kept and reported, not removed (see §8).
- **Integrated Gradients (IG).** An attribution method: it estimates how much each input
  token contributed to the prediction by integrating gradients along a path from a
  *baseline* input to the real one. Used to ask "did the model look at the same words in
  the source sentence and its translation?"

### 2.5 Machine translation and embedding similarity

- **IndicTrans2** (`ai4bharat/indictrans2-indic-indic-1B`) is the translation model. It
  translates **directly** Indic→Indic. The project never pivots through English: an
  English hop adds a second error source and makes it impossible to attribute damage.
  Decoding is frozen (beam 5, max length 256, no sampling) and identical across all nine
  directions, so quality figures are comparable between them.
- **LaBSE** is a multilingual sentence-embedding model. It maps a sentence in any language
  to a vector such that translations land close together. **Cosine similarity** between two
  vectors (1.0 = same direction) is the project's measure of "did the meaning survive". The
  threshold below which a translation is called **drifted** is **τ (tau)**.
- **What LaBSE similarity is and is not.** It is a good *relative* ranking of translation
  directions. It is a poor *absolute* quality score: machine translation scores **higher**
  than human translation of the same sentence, because the metric rewards literal wording
  and a human translator paraphrases. LaBSE similarity also falls as text gets shorter, which
  is why τ is different per task.

---

## 3. The research question and why the design looks the way it does

### 3.1 The original design

Take three native source splits (Hindi `H`, Bengali `B`, Telugu `T`). Translate each into
the other three languages, giving **9 translation directions** and 12 splits per task, in
three **blocks**. A block is 4-way parallel: one native split plus three MT versions,
sharing item IDs and gold labels.

| Block | Native | MT targets |
|---|---|---|
| `H` | Hindi | Bengali, Malayalam, Telugu |
| `B` | Bengali | Hindi, Malayalam, Telugu |
| `T` | Telugu | Hindi, Bengali, Malayalam |

This enables comparisons that cannot otherwise be made:

- **Transfer cells.** Train on `H_nat`; evaluate on Bengali, Telugu, Malayalam.
- **Translationese.** Evaluate on native Hindi (`H_nat`) vs Hindi that was machine-translated
  from Bengali (`Hi←B`) vs from Telugu (`Hi←T`). Same language, same labels, different
  *provenance*. That isolates the penalty of text being translated at all from the penalty
  of the language being different. It is the experiment that tells transfer failure apart
  from translation damage.

### 3.2 The audit that changed everything (T-102)

The design assumed the three native splits were *independent* text. An audit tested this
properly and found they are not, for two of the three tasks:

- **Tasks 2 and 3 are the same sentences, translated by humans, in shuffled order.** Task 3 is
  proven parallel by its `URL` column. Task 2 was shown by embedding match: 88–93% of
  sampled sentences have a near-duplicate in another language above τ, against 0.0% for
  different content in the same domain.
- **Task 1 is genuinely independent** (median nearest cross-language similarity 0.58–0.63).
  It is the only uncontaminated task, but it has no class label.

How the test was built matters, because the first idea (embed 200 random cross-language
pairs and look at the distribution) tests nothing: random pairs score low whether or not
the corpora are parallel. The finding rests on three tests plus controls:

1. **Structural** (free): row counts and class proportions. Near-identical shares across
   "independent" corpora indicate a shared source.
2. **Order-aligned**: cosine of row *i* in one language against row *i* in another vs a
   shuffled control. Catches parallel corpora that kept row order.
3. **Nearest-neighbour**: for each sampled sentence, the best match over the **whole**
   other split. This still fires when rows were shuffled, and is the one the written
   finding cites.
4. **Controls**: a positive anchor (a task known to be parallel) and negative anchors
   (same domain, genuinely different content). The result is only valid if negatives sit
   far below τ.

### 3.3 Consequences — why Phase 1 is shaped as it is

For tasks 2 and 3 the same sentence exists in Hindi, Bengali and Telugu natively. That has
two big effects.

**A leak risk.** If you train on Hindi item 17 and "evaluate" on Bengali item 17 (the human
translation of the same sentence), you are testing on the training sentence in another
script. So:

- An **alignment step** (T-102b) recovers which row in each language is the same item:
  exact join on `URL` for task 3 (532/532 align); mutual nearest-neighbour on LaBSE above τ,
  required to be consistent three ways, for task 2 (1,769 items align).
- Every transfer condition evaluates only on items from a **disjoint, seeded, hash-based
  partition** of item IDs (≈50/50 train/eval). Hashing the ID means every language agrees
  on which half an item is in, with no coordination, and it survives reordering. This is
  T-113, and it is mandatory for tasks 2 and 3. Task 1 needs none: its blocks hold genuinely
  different sentences.

**A free quality ceiling.** For tasks 2 and 3, a *human* translation of every item already
exists in the other native languages. Scoring MT against it gives a measure of what "a good
translation" scores, which makes τ defensible instead of asserted. Malayalam and task 1 have
no such reference and fall back to source similarity only; the reports say so.

### 3.4 Why tasks 2 and 3 carry the classification phases, and task 1 does not

Phases 2–7 need a class label to compute a classifier's accuracy, so they use **tasks 2 and
3 only**. Task 1 (numbers) drives the translation-quality analysis in Phase 1: do numerals
survive Indic→Indic translation? Roughly one financial numeral in six does not (§9.1).

---

## 4. The data: what is in the corpus

### 4.1 Layout

```
data/base_paper/raw/task_{n}/{hindi,bengali,telugu}.xlsx   upstream, byte-for-byte, SHA-256 in manifest.json
data/raw/task_{n}/{H,B,T}/{lang}.parquet                    corpus splits before freezing
data/v1.0/task_{n}/{H,B,T}/{lang}.parquet                   FROZEN release; read-only; manifest.json with hashes
data/v1.0/DATASHEET.md                                      generated from artefacts
data/verification/task_{n}/                                 alignment maps, LaBSE scores, embeddings (.npy, gitignored)
data/predictions/task_{n}/[encoder/]{condition}.parquet     one row per (item, seed) prediction     (Phase 3+)
data/failures/task_{n}/[encoder/]{condition}.parquet        the subset: right in source, wrong in target
data/diagnostics/task_{n}/[encoder/]{condition}.parquet     one cause label per failure             (Phase 6)
data/diagnostics/audit/                                     every module's verdict per row
reports/                                                    generated reports (.md + .parquet)
experiments.csv                                             append-only run log (the one sanctioned CSV)
checkpoints/{run_id}/                                       model weights (gitignored, regenerable)
```

The `[encoder/]` level is omitted for `indicbert-v2` (its artefacts predate the second
encoder) and present for `mbert-base`. `task_{n}` exists in every derived path because
tasks 2 and 3 produce identically named conditions and an `item_id` is only unique within a
task. Writing them together once silently merged two tasks' results; the path level
prevents a repeat.

### 4.2 Row schema (classification tasks)

| Column | Meaning |
|---|---|
| `block_id` | `H`, `B` or `T` |
| `item_id` | content-derived stable key, e.g. `t2_edb0fc5f843f` |
| `lang` | `hin` `ben` `tel` `mal` |
| `origin` | `native` or `mt` |
| `src_lang` | language it was translated from; null if native |
| `text` | the sentence |
| `label`, `label_id` | gold class, and its index in `labels.json` |
| `labse_sim` | similarity to the source sentence; null for native rows |
| `flags` | list of strings, e.g. `translation_drift`, `placeholder_leak`, `escape_leak`, `script_leakage`, `truncation_suspect`, `unaligned` |

The primary key is `(block_id, item_id, lang)`. Task 1 replaces `label`/`label_id` with
`number_indic`, `number_english`, `start_posn`, `end_posn` (end-exclusive character offsets),
`magnitude` and `span_recovered`.

**IDs are derived from content, never from row position**, so an upstream reorder cannot
silently re-point an ID at a different sentence. In task 1 an "item" is a *(sentence, span)*
pair, so a sentence with three numbers is three items. Exact duplicate rows get an
occurrence suffix (`#1`, `#2`) because they must not be dropped and a content hash cannot
tell them apart.

### 4.3 A real row pair

A Hindi native row in block H (task 2, label `unsustainable`) and its Bengali MT row:

```
hin (native)  सुरक्षा के लिहाज से सबसे कम प्रभाव वाले नेटवर्क फॉल्ट (48%) और लाइटिंग प्वाइंट आउट (25%) ...
ben (mt)      নিরাপত্তার দিক থেকে সবচেয়ে কম প্রভাবের নেটওয়ার্ক ফল্ট (48%) এবং লাইটিং পয়েন্ট আউট (25%) ...
              origin=mt  src_lang=hin  label=unsustainable  labse_sim=0.972  flags=[]
```

The two percentages survive, the label is carried over unchanged, and similarity is far
above τ = 0.82.

A task-1 example shows span recovery. Hindi `एनपीसीआई द्वारा 24 घंटे संदाय प्रणाली` has the
number at characters 16–18. Its Telugu MT `ఎన్పీసీపీ ద్వారా 24 గంటల చెల్లింపు వ్యవస్థ` has it at 17–19.
**Offsets do not carry across a translation**, so they are *re-found* in the output, and
`span_recovered=True` records that it worked. Where the number is gone, offsets are `-1`
and the row is flagged `span_not_recovered`. The row stays either way.

### 4.4 Size

| | native rows | MT rows | frozen total |
|---|---|---|---|
| task 1 | 22,786 | 68,358 | 91,144 |
| task 2 | 6,538 | 19,614 | 26,152 |
| task 3 | 1,596 | 4,788 | 6,384 |
| | | **92,760** | **123,680 in 36 splits** |

---

## 5. The seven phases, one at a time

For each: what it does, why, the non-obvious part, and where the details are.

### Phase 1 — Parallel corpus (T-101 … T-114)

**Pipeline in order:**

| Task | What | Notes worth knowing |
|---|---|---|
| T-102 | Audit native splits; test independence | §3.2. Exits non-zero for tasks 2 and 3 *by design*: that is the recorded finding |
| T-102b | Recover item correspondence | Refuses task 1 (aligning it would invent a correspondence) |
| T-103 | ID scheme, label schema, Parquet IO | `corpus_io.py` is the only module allowed to read/write corpus Parquet |
| T-104 | IndicTrans2 wrapper, frozen decoding | Resumable, batches adapt to GPU memory; a 20-sentence/direction hand-check |
| T-105 | Numeral and entity preservation checker | Compares *values*, handles scale words; excludes corrupted rows |
| T-106 | Generate all 9 directions × 3 tasks | ~14 GPU-hours for task 1; identical sentences translated once |
| T-107 | Structural integrity, script leakage, corruption flags | Blocking checks vs non-blocking findings |
| T-108 | LaBSE similarity for **every** MT pair | Also scores MT against the human reference where one exists |
| T-109 | Per-direction drift report | Below τ ⇒ flag `translation_drift`, keep the row |
| T-110 | Calibrate τ per task | Task 3 uses 0.73, not 0.82 |
| T-111 | Rank the 9 directions | Mean of ranks, deliberately not a weighted sum |
| T-112 | Freeze `data/v1.0/` | SHA-256 manifest, read-only, tamper-tested |
| T-113 | Evaluation-condition matrix | Item-level partition; native text preferred as eval |
| T-114 | Datasheet | Every figure read back from artefacts; `--check` fails if stale |

**The two silent corruption modes** (found in T-104, §3.4 of CLAUDE.md). Both leave
fluent-looking text and are invisible without an explicit check:

1. **Placeholder leakage.** The pre-processor swaps entities for placeholders (`<ID 1>`) and
   restores them after decoding. The model sometimes *translates the placeholder itself*
   (`ID` → `আই. ডি.`), so restoration fails and the real content (an `@handle`, a numeric
   range) is lost.
2. **Escape leakage.** A nukta character in the source can come out as the literal text
   `u09bc` transliterated into the target script.

Real rate on task 1's sample was 3.2% overall, not the ~1% first estimated, because the
first detector covered only two of four target scripts. The lesson recorded in the spec: a
checker has to be validated against real output, and one that raises false alarms is as bad
as one that misses things.

**The model decision, as an example of project culture.** The 320M distilled model was
chosen first (8× faster, only 3.3 points behind on "digits identical"). T-105 then measured
*corruption* directly: 320M mangled 10.0% of rows against the 1B's 4.4%. The proxy metric
could not see that. The 1B was selected. That reversal is why the project distrusts proxy
metrics.

**τ calibration.** The same 0.82 rejects 0.0% of human translations on task 2 but 23.4% on
task 3 (headlines: median 86 characters vs 148). Same threshold, different meaning. So τ is
per task, set where 95% of human translations pass. Task 1 has no human reference, so its
0.82 is explicitly "inherited, not calibrated".

**Task-1 span recovery** (the headline of Phase 1): 82.8% recovered. Of the unrecovered
17.2%, 9.7% have digits present but the annotated value gone, 3.7% are corrupted, 3.3% have
no digits at all, 0.4% are the same quantity rescaled, and **only 0.04% (30 rows) are the
recovery code missing a number that was actually there.** So about one financial numeral in
six genuinely does not survive Indic→Indic translation.

Details: `docs/phase1.md`, `CLAUDE.md`, `data/v1.0/DATASHEET.md`.

### Phase 2 — Training pipeline and Hindi baseline (T-201 … T-208)

Fine-tune IndicBERT-v2 on native Hindi for tasks 2 and 3, three seeds each, and prove it
works.

- **T-203** is the training loop; a smoke test fits 48 rows to train F1 = 1.0, which proves
  the wiring. Determinism needed a fix: the classifier head is randomly initialised, so the
  seed must be set *before* constructing the model.
- **T-204** is the YAML config system. Every run records a **config hash** covering
  hyperparameters, data split and dev/test fractions. Unknown YAML keys are rejected.
- **T-205** measures GPU memory. The machine has **one RTX 3050 laptop GPU with 4 GB**. This
  limit shapes the whole project (see §10).
- **T-207 — the gate.** Compare in-language macro-F1 with IndicFinNLP's published number:
  task 2 ours 0.823 vs published 0.86 (below, diagnosed as a smaller resampled test fold with
  no paper split); task 3 ours 0.153 vs 0.05 (above). Nothing in Phase 3 may start until it
  passes. The reasoning: if the baseline was never trained properly, "the model transfers
  badly" and "the model was broken" are indistinguishable.
- **T-208** is checkpoint naming (`checkpoints/<run_id>/`, `run_id = <config-stem>_seed<N>`),
  retrieval and pruning. It also caught a bug where two different config hashes were being
  compared and every re-run discarded a perfectly resumable checkpoint.

Task 3 is at a **data ceiling**: ~370 training rows over 10 classes. The published baseline
is 0.05 (essentially one-class prediction), and the paper's own remedy, paraphrase
augmentation to 4,774 rows, is out of scope.

### Phase 3 — Zero-shot transfer evaluation (T-301 … T-308)

Run frozen checkpoints over target splits and record every individual prediction.

- **T-301** loads a checkpoint with all parameters frozen and *asserts* no gradient is
  built. Zero-shot is enforced, not promised.
- **T-302** writes a per-instance log: `item_id, gold, pred, probs, seed, …`. This is a
  *deliverable*, because Phase 6 cannot be rebuilt from aggregate metrics.
- **T-304** gives bootstrap CIs and the transfer gap, plus `within_seed_noise()` to make
  "is this a finding" mechanical.
- **T-305** runs all conditions × 3 seeds. Two real leaks were found and fixed here: a
  `partition:eval` filter that was not actually applied (so the model was evaluated on
  items it trained on, in another language), and a second script that overwrote
  correctly-filtered logs with unfiltered ones.
- **T-306 — the mismatch filter.** An item is a **failure** iff `pred_source == gold` AND
  `pred_target != gold`, joined on `(item_id, seed)`. The model got this exact sentence
  right one language earlier, so "the model is incapable" and "this item is hard" are ruled
  out by construction. This predicate defines what Phases 6 and 7 study.
- **T-307** exports the failures. **T-308** renders the tables.

Result: the gap is 0.39–0.45 on task 2 and 0.50–0.51 on task 3 and is "not within seed noise"
everywhere measured. Evaluating on native vs MT target text barely changes it.

### Phase 4 — More source languages (T-401 … T-408)

Train on Bengali and Telugu too (not only Hindi), so that "how well does Hindi travel"
becomes "which language is the best one to start from".

- The spec's goal was that this be **configuration, not code**. T-401 verified it: nothing
  was hardcoded to block H. It did find one missing guard: a parity check that only
  compared a config with the same-*language* config, so the Bengali config had nothing to be
  compared against. `check_block_parity` now compares against the same-*encoder* Hindi config
  and permits only `lang` to differ.
- T-402/T-403 are gates again (Bengali, Telugu baselines vs published numbers). Both pass,
  with the task-3 caveat that the published bar is just chance.
- T-408 produces `reports/source_comparison.md`.

Findings (both encoders): **Bengali is the best source, Telugu second, Hindi worst.**
Crossing into Dravidian costs more than crossing out of it. Hindi→Bengali loses more than
Bengali→Hindi (0.4477 vs 0.2631 on IndicBERT-v2, task 2) with in-language ceilings differing
by 0.002. Two things did *not* replicate across encoders: that typological proximity
predicts the best source, and the Bengali↔Telugu asymmetry. Task 3 supports none of it.

### Phase 5 — Second encoder (T-500 … T-506)

Repeat the whole protocol with mBERT. T-500 is a smoke test that first found *five*
places where "model-agnostic" code was not (checkpoint names with `indicbert` pasted in,
prediction logs lacking `encoder_id`, un-namespaced paths). Two of the five would have
silently mixed encoders' results.

XLM-R was dropped on VRAM grounds. Adapter training (MAD-X) was considered and rejected: it
would change *how* the model was trained and so destroy the comparison it was meant to
rescue.

Result: **mBERT beats IndicBERT-v2 in every comparable cell** (task 2 in-language 0.874 vs
0.823; task 3 0.335 vs 0.153). Why is not established. mBERT is bigger *and* more
multilingual, and this design cannot separate them. The decisive control, mBERT vs XLM-R,
which have the same-size 85.6M bodies, is the arm VRAM blocked, so `reports/capacity_dilution.md`
records the question as open.

### Phase 6 — Failure diagnosis (T-601 … T-609)

Give every failure exactly one cause label. The **order of checks is the design**:

```
Stage 1  semantic gate        LaBSE sim < τ  ─────► translation_drift   (terminal: goes no further)
Stage 2  linguistic modules   (only semantically-equivalent pairs reach here)
            tokenizer_fragmentation → morphological_masking → terminology_gap
Stage 3  data / model         (claim only what Stages 1–2 left unexplained)
            provenance_specific → cross_lingual_hard → class_collapse → seed_unstable
         nothing fired  ───────► unattributed
```

Why gate first: if the linguistic modules ran first, noise from *our own* translation would
be mis-blamed on tokenization or morphology.

| Label | Fires when |
|---|---|
| `translation_drift` | LaBSE similarity below τ |
| `tokenizer_fragmentation` | the target splits into ≥ 2.0× as many tokens as the source (`R_frag`, using the encoder's own tokenizer) |
| `morphological_masking` | an ESG lexicon term is present but buried under inflection (per-language rules; Hindi can never fire) |
| `terminology_gap` | Integrated Gradients saliency diverges across the source/target pair beyond a per-cell threshold |
| `provenance_specific` | the same (item, seed) is **right** on the twin condition (native vs MT target, same language and source) |
| `cross_lingual_hard` | the same (item, seed), same source, fails in **all three** target languages |
| `class_collapse` | the run predicts one class on ≥ 75% of the condition's target rows and this failure is that class |
| `seed_unstable` | the same item is right on another seed |
| `unattributed` | nothing fired |

Each label belongs to a level-1 group (`linguistic`, `data`, `model`, `unattributed`),
always derived from the level-2 label so they cannot disagree. Every module that fired is
recorded in an **audit trail**, so you can see what else matched besides the winner.

Things you must know before trusting these labels:

- **Three modules were validated and found weak.** Masked terms do not raise failure rates
  (masked fails more often than intact in only 20 of 44 cells). Saliency divergence does not
  separate failures from correct pairs (AUC 0.48–0.52). The orthographic/numeral module was
  **removed** because digit-system mismatches are *not* over-represented among failures
  (e.g. 12% of failures vs 34% of items). The two weak labels were kept on purpose for the
  demonstration, and the limitation travels with them everywhere they are reported.
- **Thresholds are never tuned to improve the distribution.** The saliency threshold was set
  at the 95th percentile of divergence on pairs the model got *right* (a null), fixed
  *before* looking at failures. Reading it off the failures would fix the firing rate by
  construction.
- **The Stage 3 labels are facts read off the prediction logs**, not hypotheses, so they
  need no null check. But they are generic. `seed_unstable` says only that training
  randomness decided the item.
- A bug fix worth remembering: the original morphology match used edit distance ≤ 2, so
  two-character terms matched any two-character word; 96% of 10,642 masking labels were
  fuzzy false hits. The corrected rule accepts only exact matches, one dropped final
  codepoint, or one added vowel sign (10,642 → 2,484 labels).
- **Hand spot-check:** 197/200 = 98.5% agreement, but only for task 2 + IndicBERT-v2, and
  160 of those 200 rows carry data/model labels that are read off logs anyway; the
  linguistic labels rest on 40 rows.

### Phase 7 — Root-cause findings (T-701 … T-707)

Count the labels and ask what explains the effects of Phases 3–5. The unit is the **72
cells**: 2 tasks × 2 encoders × 18 conditions (9 source→target pairs, each on native text
and on MT text).

The statistical rule is fixed *before* any table is read: **two groups differ only if their
bootstrap 95% CIs do not overlap AND the difference points the same way on both encoders.**
Anything else is reported as "one encoder only", "within noise" or "opposite directions".
Bootstrap resampling is over (condition, item) clusters, not rows, because one item failing
on three seeds is one observation.

Outputs: `reports/phase7/*.{md,parquet}` and `reports/findings.md`, generated and
`--check`-verified (it fails if the committed report drifts from its sources). Every claim
cites a row ID such as `R702-03`.

---

## 6. How the artefacts flow through the repo

```
IndicFinNLP .xlsx ──T-103 ingest──► data/raw/task_n/{H,B,T}/{nat}.parquet
        │                                      │
        │            T-102b alignment          ▼
        │            (tasks 2,3)       T-104/106 IndicTrans2 (cache/translate checkpoints)
        │                                      │
        │                                      ▼
        │                      data/raw/ ... + MT splits
        │                                      │
        │      T-105 numerals · T-107 integrity · T-108 LaBSE · T-109 drift flags
        │                                      │
        │                         T-110 τ  ·  T-111 ranking
        │                                      ▼
        └────────────────────────► T-112 FREEZE ──► data/v1.0/ (immutable)
                                               │
                  configs/eval_conditions.json │ (T-113)
                                               ▼
            Phase 2 train ──► checkpoints/ ──► Phase 3/4/5 inference
                                               │
                                  data/predictions/ ──T-306──► data/failures/
                                                                   │
                                                  Phase 6 router (LaBSE + tokenizers + IG + logs)
                                                                   │
                                                          data/diagnostics/
                                                                   │
                                                      Phase 7 counts ──► reports/findings.md
                                                                   │
                                           dashboard/ (read-only) ◄┘
```

A downstream phase never reaches back and edits an upstream artefact. Phase 7's spec makes
this explicit: if a label looks wrong, report the rows; do not relabel.

---

## 7. Code map

**One module per concern, flat in `src/`; thin CLIs in `scripts/`.** Real logic never lives
in a script or a notebook. Each `src/` module has a matching `tests/test_*.py`.

### Phase 1 — corpus

| Module | Responsibility |
|---|---|
| `download_dataset/` | fetch upstream via kagglehub, write SHA-256 manifest (only package; exempt from `corpus_io` rule) |
| `unicode_ranges.py` | script and digit tables, danda exclusion — single source of truth |
| `audit.py`, `align.py` | independence tests; item-correspondence recovery |
| `ids.py`, `corpus_io.py` | key construction; **all** corpus Parquet reads/writes, schema and label enforcement |
| `translate.py` | IndicTrans2 wrapper, resumable, adaptive batching, deadline per batch |
| `entities.py`, `spans.py` | value-based numeral/currency check; re-locating task-1 numbers in MT output |
| `generate.py` | native split + translations → §6 MT rows with parity checks |
| `integrity.py`, `labse_gate.py` | structural/leakage checks; embeddings and cosine |
| `conditions.py`, `freeze.py`, `datasheet.py` | condition matrix + validator; hash/manifest; datasheet generator |

### Phases 2–5 — models

| Module | Responsibility |
|---|---|
| `config.py`, `experiments.py` | frozen `RunConfig` + hash; append-only `experiments.csv` |
| `data.py` | encoder registry, `load_split`, seeded stratified split, tokenisation |
| `train.py`, `metrics.py`, `vram.py`, `checkpoints.py` | the loop; macro-F1/confusion; memory probes; run-id retrieval |
| `baseline_gate.py` | the T-207/402/403 gate against published numbers |
| `inference.py`, `predictions.py` | frozen prediction with asserts; per-instance logs |
| `evaluate.py`, `transfer.py`, `mismatch.py`, `failures.py`, `results_tables.py` | run conditions; bootstrap + gap; the failure predicate; export; tables |
| `source_comparison.py`, `encoder_comparison.py`, `capacity.py` | Phase 4/5 comparisons |
| `env_check.py` | Python 3.11 + stack sanity check |

### Phases 6–7 — diagnosis

| Module | Responsibility |
|---|---|
| `diagnostics.py` | router, precedence, audit trail |
| `fragmentation.py`, `morphology.py`, `saliency.py` | `R_frag`; lexicon/suffix matching; Integrated Gradients (Captum) |
| `behaviour.py` | Stage 3 data/model labels, run per (task, encoder) cell |
| `convergence.py` | IG convergence checks |
| `findings.py` | shares, bootstrap, the "held / one encoder / noise" rule, caveat attachment |

`dashboard/` is a read-only Streamlit app (Transfer, Causes, Explorer pages). `configs/`
holds frozen settings (`translation_config.json`, `verification_config.json`,
`eval_conditions.json`, `labels.json`, `esg_terms.json`, `published_baselines.json`,
`train/*.yaml`).

---

## 8. Engineering rules and the reasons behind them

These recur across every phase. Knowing why makes the code's odd-looking defensiveness
make sense.

| Rule | Why |
|---|---|
| **Never drop a row; flag it.** | Silent row loss corrupts every downstream number and cannot be undone. Below-τ pairs are a *research category*, not garbage. Same in Phase 7: failures are never dropped; `unattributed` is reported. |
| **Parquet only, never CSV/pickle for corpus data.** | Indic text and numerals break under CSV quoting. `experiments.csv` is the single exception: it is a run log, not corpus data. |
| **All corpus IO through `corpus_io.py`.** | One place enforces schema and labels. |
| **Frozen corpus is immutable.** | `data/v1.0/` has hashes; changing a file fails verification. New versions get a new directory. |
| **Never invent, remap or reorder labels.** | `labels.json` order defines `label_id`. |
| **Never pivot through English.** | A second error source would make drift attribution meaningless. |
| **Determinism + config hash everywhere.** | Two runs of one config must be byte-identical; hashes say what produced an artefact. |
| **IDs derived from content.** | Positional IDs silently re-point if upstream reorders. |
| **Report negative and null results.** | "No difference" is an answer; do not slice until something separates. Thresholds are never tuned to produce a tidy distribution. |
| **Gates before dependent work.** | T-207/402/403 block transfer claims until baselines are validated. |
| **Validate checkers against real output.** | The numeral checker's first version reported 68–94% preservation; the bugs were in the checker. |
| **Generate reports from artefacts, with `--check`.** | Typed-in numbers drift silently; the datasheet and findings report fail a test if stale. |
| **Resumable long jobs; long jobs are run by the user.** | GPU access is intermittent. Hours-long jobs checkpoint after every batch. |
| **`transformers` pinned to 4.57.6.** | IndicTrans2's remote model code breaks on 5.x in ways that cannot be reproducibly patched. `use_cache=False` is also mandatory. |
| **Refuse silent CPU fallback.** | CPU runs float32 vs the GPU's float16, so the same sentence translates differently while everything looks normal. |
| **Python 3.11 exactly**, Linux/WSL2 only. | Asserted at entry points. |

---

## 9. What was found, and how far to trust it

### 9.1 Translation quality (Phase 1)

- Roughly one financial numeral in six does not survive Indic→Indic translation intact
  (span recovery 82.8%).
- Corruption (placeholder and escape leaks) concentrates in Telugu-source directions for
  placeholders and Bengali-source for escapes.
- **Into-Dravidian translation drifts more than out of it**, in all three tasks, by 2.6–4.8
  points. This is the finding the project treats as most solid, because two independent
  measurements agree: the MT drift (Phase 1) and the classifier transfer gap (Phase 4).
- Hindi-source vs Bengali-source quality differs a lot on task 1 (median similarity
  0.906–0.937 vs 0.844–0.854).
- Questions like "is Hi→Ml worse than Bn→Ml?" or "are Dravidian sources worse?" get
  *different* answers on different tasks, so they are not language-pair properties.

### 9.2 Transfer (Phases 3–5)

- Zero-shot transfer loses about half of accuracy; native vs MT test text hardly changes it.
- Source ranking: Bengali < Telugu < Hindi in cost (lower is better), both encoders.
  Task 2 average accuracy lost leaving each source (IndicBERT-v2 / mBERT): Bengali
  0.34/0.15, Telugu 0.37/0.17, Hindi 0.43/0.20.
- mBERT beats IndicBERT-v2 everywhere, with the cause unresolved.
- Task 3 supports no source comparison at all (every gap is 0.47–0.53).

### 9.3 Failure causes (Phase 7)

| Task / encoder | unattributed | linguistic | data | model |
|---|---|---|---|---|
| task 2 / IndicBERT-v2 | 9.2% | 12.0% | 44.6% | 34.2% |
| task 2 / mBERT | 19.7% | 11.3% | 48.3% | 20.8% |
| task 3 / IndicBERT-v2 | 11.8% | 8.8% | 72.2% | 7.1% |
| task 3 / mBERT | 14.4% | 9.4% | 64.9% | 11.2% |

- Most failures are described by *pattern* (`cross_lingual_hard` is the largest label
  everywhere) and not by a language mechanism.
- IndicBERT-v2 task-2 failures include 27.3% `class_collapse`; mBERT's are 0%.
- Our own translation is not the main cause: 9.5–25% of failures on MT text carry
  `translation_drift` or `provenance_specific`.
- A trap: failures going *into* Telugu get `morphological_masking` far more often than
  failures coming out. That difference rests on Hindi targets, where the module can never
  fire. Comparing Bengali↔Telugu, with no Hindi, it does not hold. `findings.CAVEATS`
  attaches such caveats automatically.

**How to read a share.** A share says what the *failures* are made of. It does not say a
label makes failure more likely, because labels were never compared against correct
predictions.

---

## 10. Limitations and open questions

1. **No native Malayalam.** Malayalam is always MT; its "native" evaluation uses MT from a
   different block.
2. **No native-speaker review.** A maintainer checked 20 sentences per direction (0.3% of
   task 1's MT output, 0.9% of task 2's, 3.8% of task 3's). Nobody on the project reads
   Bengali, Telugu or Malayalam. Automated checks establish numerals, structure, labels and
   script survive, not that every sentence means the right thing.
3. **LaBSE rewards literalness.** Use `labse_sim` for ranking, not as an absolute score.
4. **Tasks 2 and 3 are not independent across languages.** That is why the item partition is
   mandatory.
5. **Task 3 is at its data ceiling**, and 3 of 9 IndicBERT-v2 task-3 runs never trained.
6. **Only two encoders**, because of the 4 GB GPU. Whether breadth or capacity explains
   mBERT's lead is recorded as open.
7. **Diagnostic modules are weakly validated** (§5, Phase 6). Hand check covers one of four
   (task, encoder) cells; task-3 IndicBERT-v2 has 8.2% unconverged saliency rows, above the
   5% limit.
8. **Bengali native vs MT differ in digit system as well as provenance.**
9. **Hindi targets cannot receive `morphological_masking`.**
10. **Task 1 has no classifier**, so it has no transfer or diagnostic results.
11. **`tokenizer_fragmentation` essentially never fires** (0.0–0.5% of failures); the
    threshold was not lowered to change that.

---

## 11. Running things

```bash
# Python 3.11 venv; from the repo root, always as modules
python -m src.env_check                      # interpreter, stack, CUDA

# Phase 1 (examples; translation jobs are hours long and run by the maintainer)
python -m scripts.t102_audit --task 2 --independence
python -m scripts.t106_generate --task 3 --dry-run
python -m scripts.t112_freeze --task 3 --verify
python -m scripts.t114_datasheet --check

# Phase 2–5
python -m scripts.t206_baseline              # train, resumable
python -m scripts.t207_gate
python -m scripts.t305_run_conditions

# Phase 6–7 (Phase 7 is CPU-only, ~30 s)
python -m scripts.t601_diagnostics
python -m scripts.t707_findings --check

# Dashboard
streamlit run dashboard/app.py
```

Gated Hugging Face models (IndicTrans2, IndicBERT-v2) require accepting each licence and
`hf auth login`. When running sweeps use `HF_HUB_OFFLINE=1`. Tests: `pytest` (slow tests
need cached gated models). Check each script's `--help`; flags above come from the specs.

---

## 12. Glossary

| Term | Meaning here |
|---|---|
| **Block** (`H`/`B`/`T`) | One native source split plus its three MT translations; 4-way parallel |
| **Native / MT** | Human-written upstream text / IndicTrans2 output |
| **Direction** | One source→target translation, e.g. `hin→tel`; nine of them |
| **Cell / condition** | A (source, target, text provenance) evaluation. 21 per task = 9 native transfers + 9 MT transfers + 3 translationese; Phase 7 uses the 18 transfer ones per (task, encoder) |
| **Translationese** | Same language and labels, different provenance (native vs MT-from-another-language) |
| **Quadrant** | Family pair: IA→IA, IA→Dr, Dr→IA, Dr→Dr |
| **Transfer gap** | In-language score minus target score, on the same items |
| **Failure** | Item right in source language, wrong in target (same seed) |
| **τ (tau)** | LaBSE cosine threshold below which an MT pair is `translation_drift`; per task |
| **Drift** | Translation whose meaning moved: similarity below τ |
| **Partition** | Seeded hash split of `item_id` into train/eval halves |
| **Ceiling** | The model's in-language score on its own source language |
| **Unattributed** | A failure no diagnostic module explained; reported, never hidden |
| **Span recovery** | Re-finding a task-1 number in MT output; `span_recovered` records success |
| **Nukta** | Dot below a consonant; source of Unicode-normalisation and escape-leak bugs |
| **Danda** `।` | Indic full stop, excluded from script tallies |
| **Lakh / crore** | 10⁵ / 10⁷ |
| **ESG** | Environmental, Social, Governance |
| **IG** | Integrated Gradients attribution |
| **R_frag** | Target tokens ÷ source tokens, with the encoder's own tokenizer |
| **Config hash** | Fingerprint of everything that can change a run's result |
| **Freeze** | Make `data/v1.0/` read-only and hash-manifested |

---

## 13. Where to read next

Suggested order for a first deep read:

1. This document.
2. `README.md` — the summary, results and limitations.
3. `data/v1.0/DATASHEET.md` — what the corpus is.
4. `docs/phase1.md` — corpus construction in detail.
5. `docs/phase3.md`, `docs/phase4.md`, `docs/phase5.md` — transfer, sources, encoders.
6. `CLAUDE4.md` (Phase 6) then `CLAUDE7.md` (Phase 7) — the diagnostic design and counting rules.
7. `reports/findings.md` — the end result, with every claim traced to a row ID.
8. `docs/implementation_results.md` — implementation-level results across phases.

When a number in any document looks surprising, the generated report in `reports/` is the
authority; the specs are history plus rationale.
