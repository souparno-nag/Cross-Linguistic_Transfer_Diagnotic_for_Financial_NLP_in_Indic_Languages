# Phase 1 — Parallel Corpus Construction

**Specification:** `CLAUDE.md`
**Status:** Complete. All three tasks frozen at `data/v1.0/`.

## What this phase builds

Three frozen corpora — one per financial-NLP task — where the same underlying content
exists across four Indic languages (Hindi, Bengali, Telugu, Malayalam), so a model can
later be evaluated on *the same item* in every language rather than on different content
in each. Everything in Phases 2 and 3 depends on this corpus being row-aligned,
label-preserving, and never silently changed after it is frozen: losing or corrupting a
row here corrupts every downstream result and cannot be fixed retroactively once
training and evaluation have been run against it.

## Where the data comes from

The only upstream source is **IndicFinNLP** (Ghosh et al., LREC-COLING 2024), a Kaggle
dataset (`sohomghosh/indicfinnlp-financial-nlp-for-indian-languages`) that ships three
tasks, each already available in Hindi, Bengali and Telugu — **but not Malayalam**. That
absence is upstream, not a choice made by this project, and it is why Malayalam appears
nowhere in this corpus except as a machine translation.

| Task | Content | Key upstream columns | Shape |
|---|---|---|---|
| `task_1` | Financial text with a number's character position marked | `indic`, `number_english`, `number_indic`, `start_posn`, `end_posn`, `magnitude` | Span annotation — no class label |
| `task_2` | Sustainability sentences | `sentence_indic`, `label` (`sustainable`/`unsustainable`) | Binary classification |
| `task_3` | ESG news headlines | `URL`, `news_title_indic`, `ESG_Theme` | 10-class classification |

`python -m src.download_dataset.download` copies the Kaggle release into
`data/base_paper/raw/task_{n}/{language}.xlsx` and records a SHA-256 per file in
`data/base_paper/manifest.json`. This directory is treated as immutable input — the
`.xlsx` files are the one place this project stores anything other than Parquet, because
they are upstream's own release format and are never touched after ingest.

## The corpus design — and the discovery that reshaped it

The original design assumed a clean, symmetric setup: pick one native (human-written)
source split per language block, machine-translate it into the other three languages,
and end up with three **independently sourced** blocks, each 4-way parallel:

| Block | Native source | Machine-translated into |
|---|---|---|
| `H` | Hindi | Bengali, Malayalam, Telugu |
| `B` | Bengali | Hindi, Malayalam, Telugu |
| `T` | Telugu | Hindi, Bengali, Malayalam |

That gives **9 translation directions and 12 splits per task** (3 native + 9 machine-translated).
Every language ends up with three versions:

| Language | Versions | Native version exists? |
|---|---|---|
| Hindi | `H_nat`, `Hi←B`, `Hi←T` | yes |
| Bengali | `B_nat`, `Bn←H`, `Bn←T` | yes |
| Telugu | `T_nat`, `Te←H`, `Te←B` | yes |
| Malayalam | `Ml←H`, `Ml←B`, `Ml←T` | **no** |

This design is locked and is not to be changed without deliberate sign-off — it was
decided, reversed, and re-decided once already during the project, and the working
agreement (§11 of `CLAUDE.md`) explicitly asks that nobody change it unilaterally again.

### T-102 found the premise didn't hold for two of the three tasks

The design above assumes each task's three native language splits are three
*independently written* corpora. **T-102's audit found that's only true for task 1.**
For tasks 2 and 3, the Hindi, Bengali and Telugu "native" splits turned out to be the
**same set of sentences**, professionally human-translated and shuffled into a different
row order per language — not three separate corpora at all.

This was established with three separate checks, not just a hunch:

1. **Structural check (free, no model)** — do the languages have near-identical class
   proportions? For task 3, the article `URL`s are literally shared across languages, which
   settles the question outright.
2. **Order-aligned check** — cosine similarity between row *i* of one language and row
   *i* of another, compared to a shuffled control. Catches parallel content that happens
   to still be in the same row order (this is how task 3 was built).
3. **Nearest-neighbour check (the real test)** — for each of 200 sampled source
   sentences, the *best* match anywhere in the entire target split, using LaBSE sentence
   embeddings (a model that maps sentences with similar meaning, in any language, to
   similar vectors) and cosine similarity. This catches parallel content even if rows
   were shuffled or some were dropped, which the order-aligned check cannot.

The numbers were unambiguous:

| Task | Median best-match similarity | % of sampled pairs above the parallelism threshold (τ=0.82) | Verdict |
|---|---|---|---|
| `task_1` | 0.58–0.64 | 8.5–15.5% | **Independent** — close to a same-domain, genuinely-different-content control (0.0%) |
| `task_2` | 0.88–0.91 | 88.5–93.5% | **Parallel** — same content, shuffled |
| `task_3` | proven directly by shared `URL`s | — | **Parallel** — same articles, shuffled |

Task 2's matched pairs also carried identical numeral sets 80% of the time — further
confirmation that these are the same sentences translated by a person, not different
sentences that merely resemble each other.

**Why this matters:** without knowing that tasks 2 and 3 are shuffled duplicates of one
content set, a later phase could accidentally "train" on the Hindi version of an item and
"evaluate cross-lingual transfer" on the Bengali version of the *same item* — testing
whether the model memorized the sentence, not whether it generalizes across languages.
The fix is described in T-102b and T-113 below. Task 1's three languages, by contrast,
hold genuinely different sentences, which makes it the one task where zero-shot transfer
can be measured with no extra bookkeeping — but also the one task with no independent
Malayalam-quality check available (see T-102b).

This also turned a data problem into an asset. Since tasks 2 and 3's three languages are
*human* translations of the same content, once the alignment is recovered (T-102b),
every Indic→Indic language pair among Hindi/Bengali/Telugu has a genuine human reference
translation for comparison. That reference becomes the quality ceiling used to calibrate
how strict the machine-translation quality check should be (T-110) — what a human
translator's version of a sentence scores on the similarity metric is what a machine
translation can reasonably be expected to approach. Malayalam has no such reference in
any task, and task 1 has none in any direction, so both fall back to comparing the
translation only against its own source sentence.

### Per-task summary of what this means downstream

| Task | Are the native splits independent? | Alignment step needed? | Consequence |
|---|---|---|---|
| `task_1` | **Yes** | None | The corpus design above works exactly as originally written; this is the only task where a zero-shot transfer cell can't accidentally test on training content. |
| `task_2` | No — shuffled duplicates | **Required**, by embedding-based matching | Without alignment, no way to tell whether a training item and an evaluation item are the same sentence. |
| `task_3` | No — shuffled duplicates | Free — exact join on the `URL` column | Same risk as task 2, but the join key removes any need for a model. |

## Canonical row schema

Every corpus Parquet file — for tasks 2 and 3 — has exactly these columns:

| Column | Type | Notes |
|---|---|---|
| `block_id` | str | `H` / `B` / `T` |
| `item_id` | str | stable join key, **derived from content, not row position** |
| `lang` | str | `hin` / `ben` / `tel` / `mal` |
| `origin` | str | `native` / `mt` |
| `src_lang` | str | `null` when native |
| `text` | str | |
| `label` | str | must appear in `configs/labels.json` |
| `label_id` | int | index into that label list |
| `labse_sim` | float | `null` for native rows |
| `flags` | list[str] | e.g. `translation_drift`, `entity_loss` — see below |

`(block_id, item_id, lang)` is the primary key. Task 1 replaces `label`/`label_id` with a
numeral-annotation schema instead (below), since it has no class label at all.

**Why `item_id` is derived from content and never from row position:** if the upstream
data is ever reordered, a positional ID would silently start pointing at the wrong item.
Content-derived IDs survive that. This has two consequences worth knowing before
querying the data:

- **A task-1 "item" is a `(sentence, span)` pair, not a sentence.** If one sentence
  contains three numbers, it appears as three separate rows, one per annotated number —
  6,776 of 10,640 Hindi rows share their sentence text with at least one other row. The
  item ID therefore has to include which span, not just which sentence.
- **Exact duplicate rows get an occurrence suffix** (`…#1`, `…#2`). Task 2 has four rows
  that are identical in both text and label; they are kept (never dropped, see Hard
  Rule 1 below) and the suffix is what lets a content-derived ID tell them apart.

### Task 1's schema variant

Task 1 has no class label — it marks where a number sits inside a sentence — so it
replaces `label`/`label_id` with:

| Column | Type | Notes |
|---|---|---|
| `number_indic` | str | the number as written in the source script |
| `number_english` | str | the same value in ASCII digits |
| `start_posn` / `end_posn` | int | character offsets into `text`, end-exclusive |
| `magnitude` | int | upstream's scale marker |
| `span_recovered` | bool | `null` on native rows — see below |

Upstream's offsets are exact on the original text (`text[start_posn:end_posn]` reproduces
`number_indic` on 500/500 sampled rows per language) — but **those offsets do not survive
translation**, because a translated sentence has a different length in a different
script. A machine-translated row's offsets have to be *recovered* — the numeral located
freshly in the translated output — not carried across. Recovery can fail (the model may
drop, reword, or garble the number); when it does, the row is kept and flagged rather
than discarded, because whether financial numerals survive Indic→Indic translation is
the central question this project is trying to answer, not an error condition.

## Hard rules (binding across every task)

These are treated as invariants, not style preferences — breaking any of them
invalidates the corpus for its intended use:

1. **Never drop a row.** Not for low similarity, empty output, or translation failure.
   A below-threshold pair is a *research finding to report*, not garbage to discard.
2. **Never write corpus data as CSV.** Indic scripts and financial numerals (which use
   commas as both a thousands separator and a literal character) break under CSV
   quoting rules. Parquet only.
3. **Never use pickle for corpus data.** Fine for throwaway caches, never for anything
   under `data/`.
4. **Never pivot through English.** Every translation goes directly Indic→Indic. Adding
   an English hop would introduce a second source of translation error and make it
   impossible to attribute drift to a specific language pair.
5. **`data/v1.0/` is immutable once frozen.** A new corpus version gets a new directory.
6. **Never invent, remap, or reorder labels.** They come from `configs/labels.json` and
   pass through translation unchanged.
7. **Every script takes an explicit random seed**, and two runs of the same
   configuration must produce byte-identical output.
8. **Every artifact is written with a config hash** recording exactly what produced it.
9. **Never Unicode-normalize task 1's text without recomputing its character offsets.**
   Normalizing to NFC form changes the length of certain Indic characters (it expands
   some nukta letters and contracts some Telugu vowel signs); doing so would silently
   break 880 of the 22,786 spans while leaving the text looking completely normal. Tasks
   2 and 3 carry no character offsets and can be normalized freely.

## Environment

- **Python 3.11 exactly**, asserted at every entry point.
- **Linux or WSL2** — required by the IndicTrans2 toolkit.
- `torch>=2.5`, **`transformers<5`** (pinned; see below), `numpy>=2.1`, `sentence-transformers<6`.
- Translation model: `ai4bharat/indictrans2-indic-indic-1B` (see the model-choice
  discussion under T-104/T-105 below).
- Similarity model: `sentence-transformers/LaBSE`.

### Why `transformers` is pinned below version 5

`sentence-transformers` 6.x requires `transformers>=5`, but IndicTrans2's released model
code is written against the 4.x API and breaks in three places on 5.x:

1. It imports `transformers.onnx`, which was removed in 5.x.
2. `IndicTransToolkit`'s data collator imports a class from a module path that moved in
   5.x, and the package imports every submodule eagerly, so the whole thing fails to
   import.
3. Its tokenizer sets an internal attribute *before* calling its parent class's
   constructor — 5.x's stricter attribute machinery raises an `AttributeError` for this,
   and fixing it would mean patching private internals of remote model code, which would
   not be reproducible by someone re-running this later.

The first two issues could be worked around with a shim; the third cannot be, safely.
Pinning `transformers<5` (specifically `4.57.6`, with `sentence-transformers==5.7.0`,
the newest release still compatible with it) was verified not to change any downstream
result: the alignment and independence figures reproduced bit-for-bit after the pin was
applied.

A related correctness requirement, not a performance tuning knob: IndicTrans2's decoder
must be run with `use_cache=False`, because its code checks for a legacy tuple-shaped
cache object that newer `transformers` versions no longer produce, and silently crashes
on the first decoding step otherwise. This is set in `configs/translation_config.json`.

### Hardware

Development happened on a single 4 GB laptop GPU (RTX 3050). The 1B-parameter translation
model fits (2.4 GB loaded, under 2.8 GB peak at typical settings), but everything had to
be written assuming the GPU could disappear mid-job and that a fixed batch size would
eventually hit a sentence too long to fit — see T-104/T-106 for how the pipeline handles
both.

### Getting access to IndicTrans2

The IndicTrans2 model repositories are **gated** on Hugging Face — downloads fail with a
401 error until the license is accepted for that specific repository and the machine is
authenticated (`hf auth login`, or export `HF_TOKEN`). Approval is automatic once the
license is accepted; there is no human review step to wait for.

## Storage layout

| Artifact | Format | Path |
|---|---|---|
| Upstream source | `.xlsx` (as released) | `data/base_paper/raw/task_{n}/{language}.xlsx` |
| Pre-freeze corpus splits | Parquet | `data/raw/task_{n}/{block}/{lang}.parquet` |
| Frozen corpus | Parquet | `data/v1.0/task_{n}/{block}/{lang}.parquet` |
| Alignment maps | Parquet | `data/verification/task_{n}/alignment.parquet` |
| Embeddings | `.npy` | `data/verification/task_{n}/emb/` (gitignored, regenerable) |
| Similarity scores | Parquet | `data/verification/task_{n}/labse_scores.parquet` |
| Labels / configs / manifests | `.json` | `configs/`, `data/base_paper/manifest.json`, `data/v1.0/manifest.json` |
| Reports | `.md` + Parquet | `reports/task_{n}/` |

Every one of these paths includes a `task_{n}` segment specifically because, without it,
the three tasks' Hindi splits (say) would collide on the same filename and one task's data
could silently overwrite another's.

All reads and writes of corpus data go through `src/corpus_io.py` — nothing else calls
`pandas`' Parquet functions directly on corpus data. This is the single place format,
schema, and label rules are enforced, so nothing downstream can accidentally bypass them.

## Tasks

### T-102 — Audit native splits and confirm independence
`scripts/t102_audit.py`, `src/audit.py`

Audits each task's three upstream native `.xlsx` files (row counts, class balance,
duplicates, empty rows) and runs the three independence checks described above
(structural, order-aligned, nearest-neighbour), plus calibration controls: a "known
parallel" positive anchor and "genuinely different content" negative anchors, so the raw
similarity numbers have something to be compared against rather than being judged in
isolation.

**Status: done for all three tasks.** Tasks 2 and 3 deliberately exit non-zero on the
independence check — that is the correct, informative result (their natives are not
independent), not a bug to silence.

| Task | Structural check | Independence verdict |
|---|---|---|
| `task_1` | passes | **Independent** (median 0.58–0.64, 8.5–15.5% above τ) |
| `task_2` | passes | **Parallel** (median 0.88–0.91, 88.5–93.5% above τ) |
| `task_3` | passes | **Parallel**, proven directly by shared `URL`s |

### T-102b — Align the parallel native splits
`src/align.py`, `scripts/t102b_align.py`

For tasks 2 and 3, recover which row in one language corresponds to which row in
another, since nothing in the upstream release records this directly.

- **Task 3** — exact join on the `URL` column. **532 of 532 items align across all three
  languages**, zero unmatched, with the same label everywhere.
- **Task 2** — no join key exists, so alignment uses **mutual nearest-neighbour
  matching**: two sentences are matched only if each is the other's best match under
  LaBSE similarity above τ, and only kept if this holds **consistently across all three
  language pairs** (three-way agreement). Mutual matching (rather than one-directional
  "nearest neighbour") prevents ten different source sentences all collapsing onto one
  popular target sentence. **1,769 items align three ways**, out of roughly 2,000–2,200
  candidate pairwise matches per language pair; the remainder are kept in their own
  splits (never dropped) but cannot form a 3-way-parallel item.
- **Task 1 is refused by the CLI, on purpose.** Its native splits are independently
  sourced (T-102 found no cross-language overlap), so there is no real correspondence to
  recover — attempting to "align" it would invent a relationship that does not exist.

A conflicting gold label between two supposedly-aligned rows is treated as a hard
failure (the corpus cannot claim one gold label for an item whose languages disagree);
an unmatched row is not a failure, since the splits are genuinely different sizes.

### T-103 — ID scheme, label schema, IO layer
`src/ids.py`, `src/corpus_io.py`, `configs/labels.json`

Builds the ID scheme described above, reconciles the class names used across the three
native splits into one canonical `configs/labels.json`, and implements the one IO layer
every module uses to read or write corpus data — including the one-time ingest of the
upstream `.xlsx` files into Parquet.

**Status: done.** The loader raises rather than silently coercing on any label not found
in `configs/labels.json`. A round-trip write-then-read of a fixture containing Malayalam
text, Telugu digits, a currency symbol, and a lakh/crore expression reproduces the bytes
exactly. A 4-way join (native + 3 translations) on each block returns the expected row
count with zero nulls for tasks 2 and 3 once T-106 generated their Malayalam arm; task 1's
cross-language join correctly returns zero rows, since it has no cross-language
correspondence to join on.

### T-104 — IndicTrans2 translation pipeline
`src/translate.py`, `configs/translation_config.json`

Wraps the IndicTrans2 model with its required pre/post-processing (`IndicProcessor`,
which substitutes named entities and other fragile tokens with placeholders before
translation and restores them afterward). Decoding parameters — beam size 5, max length
256, no sampling — are frozen and applied identically across all 9 translation
directions, so drift figures are comparable between directions rather than confounded by
different decoding settings.

The model is loaded once and reused across all 9 directions rather than reloaded per
direction — loading it nine times would mean nine loads of a several-gigabyte model,
which the 4 GB GPU cannot survive twice in a row without running out of memory.

Every batch is checkpointed to `cache/translate/`, keyed by item ID, so an interrupted
run resumes from where it left off rather than restarting — necessary because GPU access
during development was intermittent.

**Status: done.** Signed off by the project maintainer against a 20-sentences-per-direction
hand-check (180 rows per task). That sample is small relative to the full corpus (0.3% of
task 1, 0.9% of task 2, 3.8% of task 3) and is explicitly documented as a maintainer
spot-check, not an independent native-speaker audit — see the "Known limitations" section
of `data/v1.0/DATASHEET.md`.

#### Model choice: 1B parameters, not the faster 320M distilled variant

Two models were benchmarked: the full `indic-indic-1B` and a distilled `indic-indic-dist-320M`
that is roughly 8x faster. The 320M model initially looked like the better choice — it
was faster and scored similarly on a simple "did the digits survive" check. **That
digit-survival check was misleading.** Once actual output corruption (numbers or entities
silently destroyed while the sentence still reads fluently — see below) was measured
instead of just digit presence, the 320M model corrupted **more than twice as many rows**
as the 1B model (10.0% vs 4.4% on task 1). A digit-presence check cannot see this kind of
damage, because it doesn't ask whether the *sentence* stayed intact, only whether numbers
appear somewhere in the output. **Decision: use the 1B model**, accepting roughly 15
GPU-hours of translation time instead of 2–5, because a corpus with twice the silent
corruption rate is worth less than a slower, cleaner one — this is a direct instance of
Hard Rule 1's philosophy (never quietly lose or damage data) applied to a model choice
rather than a row-dropping decision.

#### Two silent corruption modes found in the model's output

Both destroy content while leaving the sentence looking fluent, so neither is visible
without an explicit automated check (built in T-107):

1. **Entity-placeholder leakage.** `IndicProcessor` temporarily replaces named entities
   (like a Twitter handle or a name) with placeholder tokens before translation and
   restores them afterward. Sometimes the model translates the *placeholder text itself*
   instead of leaving it alone, so restoration can no longer find it and the real content
   is lost — e.g. a source `@PIBHindi` comes back as a translated fragment where the
   handle used to be.
2. **Escape leakage.** Certain "nukta" characters (a diacritic used in Bengali,
   Devanagari and Telugu, e.g. `ড়`, `य़`) sometimes come out of the model as a literal
   escape code like `u09bc`, transliterated into whatever script the target language
   uses — so a Bengali source character surfaces as garbled text in Malayalam output
   rather than being translated.

Each affects roughly 3.2% of rows across all four target languages (an earlier estimate
of ~1% undercounted it because the first detector only covered two of the four target
scripts).

#### Adaptive batching — required, not optional, on a 4 GB GPU

A fixed batch size cannot work: one part of task 1's Telugu data has sentences over 1,000
characters long against a typical 117, and beam search's memory use scales with
`beam size × batch size × vocabulary size`, so a batch sized comfortably for typical rows
will run out of memory on the rare long one. `translate_adaptive()` halves the batch size
and retries whenever the GPU runs out of memory, down to a single row if necessary, and
only gives up if even one row alone doesn't fit. It also enforces a timeout per batch —
one real run saw a single ordinary-looking batch hang for 47 minutes inside the model's
own beam-search code for no apparent reason — and treats a timeout exactly like an
out-of-memory event: halve, retry, isolate the bad row, and keep going rather than losing
the whole run to one stuck batch.

**A related driver hazard was discovered and worked around:** after a stalled job was
force-killed, the GPU driver silently stopped reporting itself as available in the next
process, and the translation code — instead of erroring out — quietly fell back to running
on the CPU. That is worse than crashing, because CPU and GPU floating-point precision
differ slightly, meaning a corpus generated partly on each would no longer be
byte-reproducible even under an identical configuration. The code now refuses to
auto-select the CPU; using it requires an explicit `--device cpu` flag, and the device
actually used is recorded in every run's manifest.

### T-105 — Numeral and financial-entity preservation checker
`src/entities.py`

Checks, per translation direction, whether numerals, currency symbols, percentages, and
Indian numbering-scale terms (lakh/crore) survive translation — comparing by **value**,
not by literal digit string, because two genuinely equivalent quantities can look
completely different as text:

- **Scale-word equivalence.** `১০০ মিলিয়ন` ("100 million") and `10 करोड़` ("10 crore") are
  the same number (10⁸) written with different scale words; naively comparing the digit
  strings "100" and "10" would misreport this as a loss.
- **Thousands-separator ambiguity.** A comma can be a separator (`145,146` → the number
  145146) or, depending on context, part of a list of separate numbers (`145, 146, 150` —
  three numbers). Misreading the wrong one distorts the loss rate significantly: an
  earlier, naive version of this checker reported 17% numeral loss on task 1 where the
  real figure, after fixing this, is 1.7% — enough to make task 1 look like the worst
  translation direction in the whole project when it is actually among the better ones.

Corrupted rows (from the two corruption modes above) are excluded from the numeral-loss
rate specifically, because counting them as numeral loss would misattribute a translation
bug to the model's handling of numbers, which is a different failure with a different
cause.

**Status: done, validated against real translation output.** Three bugs were found and
fixed in the *checker itself* during validation, not in the model: a stray space after a
comma in post-processed text (`"50, 000"` parsing as two numbers, 50 and 0, instead of
one, 50000), two of four possible corrupted-escape forms not being recognized, and
currency symbols being compared as literal characters rather than by currency
identity (so `₹` against its Telugu-script equivalent read as a false loss). Fixing these
brought the false-positive count from 6–8 per direction down to 0–1. The project's own
framing for this: "a checker that cries wolf is the failure mode here" — a numeral
preservation checker is only useful if its failures are trustworthy.

### T-106 — Generate all 9 translation directions
`scripts/t106_generate.py`

Runs the actual translation for every direction, for every task, producing the
machine-translated splits.

Two efficiency and correctness measures matter here:

- **Identical sentences are translated only once.** Because a task-1 "item" is a
  `(sentence, span)` pair, one sentence with three annotated numbers would otherwise be
  sent to the model three times — wasting compute and risking three *different*
  translations of what should be one sentence shared across three rows. Deduplicating by
  distinct sentence text before translating cuts task 1's actual translation workload by
  41% (68,358 rows down to 40,401 distinct sentences).
- **Task 1's numeral offsets must be re-found in the translated text, not carried across**
  (see the schema section above). Recovery is done by matching the numeral's *value*
  (so a translation using Bengali digits, ASCII digits, or a grouped form like `25,000`
  are all recognized), and the outcome is recorded in `span_recovered`.

**Status: done — all three tasks generated, 92,760 rows across all 9 directions,
zero silently-empty translations.**

Total generation cost: roughly 31 GPU-hours across all three tasks (task 1 alone is
about 23 of those, since it has by far the most rows). Tasks were run in ascending cost
order — task 3 first — specifically so the entire pipeline could be proven correct
end-to-end in under two hours before committing a full day of GPU time to task 1.

**Task 1's numeral span recovery succeeded on 82.8% of rows** — meaning roughly one
financial numeral in six does not survive Indic→Indic translation with its position
recoverable. This is treated as a headline *finding* of the project, not a defect to
suppress, and it was broken down carefully to make sure it wasn't an artifact of the
recovery code itself rather than the translation:

| Outcome | Rows | Share |
|---|---|---|
| Span recovered correctly | 56,624 | 82.8% |
| Digits present in output, but the annotated value is gone | 6,618 | 9.7% |
| Output corrupted (placeholder/escape leakage) | 2,560 | 3.7% |
| No digits at all in the output | 2,234 | 3.3% |
| Same quantity, rewritten with a different scale word | 292 | 0.4% |
| Entity preserved but the recovery code simply missed the span | 30 | 0.04% |

Only the last row (0.04% of rows) would indict the recovery code rather than the
translation itself, which gives confidence the 82.8% figure reflects a real translation
phenomenon rather than a bug in how spans are located. A second finding surfaced here:
corruption is heavily concentrated in translations *out of* Telugu (500+ rows per
direction, versus 17–30 for Bengali-sourced directions) — a pattern the earlier,
by-target-language corruption count had not shown.

**One resume bug was found and fixed by inspecting the generated output, not by a test
failing.** When a translation job resumed from a checkpoint, the "which rows had a
problem" flag list came back from the Parquet checkpoint file as a different data type
(a numpy array instead of a plain list) than the code checking it expected, and the check
silently treated every resumed row as if its translation had failed — even though the
actual translated text was sitting there, fine, right next to the incorrect flag. Nothing
in the run's own summary reporting caught this, because that summary derived its numbers
from the text itself (correctly) rather than from the mismatched flag. It affected about
700 rows total across the corpus, all of which had fine translations mislabeled as
failures; they were repaired by re-deriving the flag from the actual output
(`--rebuild`), without needing to re-run the GPU.

### T-107 — Structural integrity check
`src/integrity.py`, `scripts/t107_integrity.py`

Runs a battery of structural checks over the generated corpus: every block still joins
4-way with no missing rows, no label drifted between a source row and its translation, no
encoding corruption, and specifically checks for:

- **Script leakage** — e.g. Devanagari characters showing up inside what should be a
  pure Malayalam split. In practice this turned out to almost always be an untranslated
  brand or website name left in the original script (`ওয়াটার.অর্গ` sitting inside an
  otherwise-Bengali sentence) rather than actual corruption — a translation-completeness
  issue, not a data-integrity one.
- The two corruption modes from T-104/T-105 (placeholder leakage, escape leakage).
- **Truncation** — output suspiciously shorter than expected, judged per-direction
  relative to that direction's own typical length ratio, because Malayalam output tends
  to run about 10% longer than its Hindi source while Bengali runs about 4% shorter — a
  single global threshold would flag normal Bengali output as truncated.

Checks are split into two tiers: **blocking** (the join breaks, a label drifted, or text
contains a literal replacement/mangled-encoding character) fails the run outright,
because these mean the corpus is making a false claim about a row. **Findings** (script
leakage, corruption, truncation) are recorded but do not fail the run, since Hard Rule 1
requires keeping the row regardless — these measure the quality of a specific
translation, not the structural soundness of the corpus itself.

**Status: done for all three tasks — zero blocking failures anywhere.** Every block
joins 4-way at the expected row count in all three tasks, and no label ever drifted from
source to translation. Findings, corpus-wide across all three tasks combined:

```
span_not_recovered   11,734    unaligned            3,693    placeholder_leak   3,546
span_ambiguous         2,566    truncation_suspect   1,143    escape_leak          736
script_leakage           441    span_scale_shift       299
```

Task 1 accounts for roughly ten times more findings of every type than tasks 2 or 3
combined, which fits the pattern already established: it is the only task whose text is
dense with numerals, named entities, and nukta characters, all of which are exactly what
the two corruption modes target.

### T-108 — LaBSE similarity gate
`src/labse_gate.py`, `scripts/t108_labse.py`

Computes a semantic similarity score between every source sentence and its machine
translation, for **every single translated pair in the corpus — nothing sampled** —
using LaBSE (a model trained specifically to produce comparable sentence embeddings
across languages). Similarity is cosine similarity between the two embeddings, which
ranges from -1 to 1 (1 meaning identical meaning as far as the model can tell).

**Status: done for all three tasks — 92,760 pairs scored.** A second, more informative
comparison became possible specifically because of the T-102/T-102b discovery: for tasks
2 and 3, where the aligned item in another language is a genuine *human* translation of
the same sentence, the machine translation can be scored directly against that human
reference (same language, same content, same script) rather than only against the
original-language source. Task 1 has no such reference in any direction, and no
Malayalam target has one anywhere, since Malayalam never had a human-written version to
compare against.

Task 1 (the only task with genuinely independent source content across languages) shows
a striking split by source language: translations originating from Hindi score
noticeably better (median 0.906–0.937) than those originating from Bengali (0.844–0.854),
with Telugu-sourced translations in between but with a much wider spread of bad outliers.
This pattern does not appear in tasks 2 or 3, which is consistent with task 1 being the
only task where the three blocks actually contain different underlying content rather
than three shuffled views of the same sentences.

### T-109 — Per-direction drift report
`scripts/t109_drift.py`

Breaks the T-108 similarity scores down by translation direction and by outcome class
(for tasks 2/3, the gold label; for task 1, whether the numeral span was recovered).
Rows scoring below the task's similarity threshold τ are flagged `translation_drift` and
**kept in the corpus** — never removed, per Hard Rule 1.

**Status: done for all three tasks.** The task-1 breakdown by span-recovery status turned
out to be, in the project's own assessment, the strongest internal consistency check
built during the whole phase: two completely independent measurements — one that locates
a specific digit string in the text, and one that embeds the whole sentence and measures
overall meaning similarity — agree closely on which rows went wrong, despite having no
knowledge of each other's output. For example, in the Telugu→Bengali direction, rows
where the numeral span *was* recovered have a median similarity of 0.921 with only 8.3%
falling below τ, while rows where the span was *lost* drop to a median of 0.707 with
68.9% below τ. A sentence that loses its numeral usually turns out to have lost more than
just the numeral.

### T-110 — Calibrate the similarity threshold τ empirically
`scripts/t110_calibrate.py`

The similarity threshold τ = 0.82, used throughout T-108/T-109 to decide whether a
translation is "too different from its source," was originally just inherited from
common practice elsewhere rather than derived from this project's own data. Since it is
this project's *only* automated translation-quality check, it needed to actually be
justified.

Calibration works by comparing three things on the same footing: what a machine
translation scores; what a **human** translation of the same sentence scores on the
identical similarity measure (made possible again by the T-102b alignment — the "native"
splits of tasks 2 and 3 are themselves human translations of one another); and what a
machine translation scores against that human reference directly.

**Status: done — τ is now set per task, not shared globally, and the reason why is
itself a finding.**

| Task | τ | Basis |
|---|---|---|
| `task_1` | 0.82 | inherited — no human reference exists to calibrate against |
| `task_2` | 0.82 | confirmed: correctly rejects 0.0% of genuine human translations |
| `task_3` | **0.73** | revised down: the original 0.82 rejected 23.4% of genuine human translations, which meant it was measuring something other than translation quality |

The reason the same threshold behaves so differently on tasks 2 and 3 turned out to be
**text length, not translation quality** — task 3 is short news headlines (median 86
characters) while task 2 is full sentences (median 148 characters), and LaBSE's
similarity score systematically drops as context gets shorter. A single global threshold
cannot be correct for both.

A second, more uncomfortable finding came out of this calibration: **machine
translations score *higher* than human translations of the identical sentence**, in every
language pair checked. This is because LaBSE's similarity metric rewards literal,
word-for-word correspondence, while a human translator naturally paraphrases — and the
metric penalizes paraphrasing as if it were an error. The practical consequence is that
this similarity score is useful for *ranking* which translation directions are more or
less reliable relative to each other, but should not be read as an absolute measure of
translation quality — a translation scoring well on it is not necessarily better than one
scoring less well, if the lower-scoring one happens to read more naturally.

### T-111 — Comparative translation-direction ranking
`scripts/t111_ranking.py`

Ranks all 9 translation directions by combining the drift rate (T-109) with the entity
preservation rate (T-105), using the **average of each measure's rank** rather than a
single weighted score — a drift rate and a numeral-preservation rate are not naturally on
the same scale, and inventing a specific weight to combine them would imply a precision
the data doesn't actually support. Averaging ranks only claims "worse on more measures,"
which is a claim the data can actually back up.

**Status: done for all three tasks.** The project set out to answer three specific
questions, and the results split in an instructive way:

| Question | task 1 | task 2 | task 3 |
|---|---|---|---|
| Is Hindi→Malayalam worse than Bengali→Malayalam? | No | No | **Yes** |
| Are translations *from* a Dravidian language worse than *from* an Indo-Aryan one? | No | No | **Yes** |
| Is the Indo-Aryan↔Dravidian penalty the same in both directions? | **No** | **No** | **No** |

The first two questions get *different* answers on different tasks — meaning neither
generalizes as a property of the language pair itself, and reporting either one as a
universal fact about (say) "Malayalam translation quality" would be wrong; it depends on
which task's specific content is being translated. **The third finding does generalize**:
in all three tasks, translating *into* a Dravidian language (Telugu, Malayalam) loses
more quality than translating *out of* one back into an Indo-Aryan language, for the same
language pair, by roughly 2.6 to 4.8 percentage points. That asymmetry — the direction of
travel mattering, not just which two languages are paired — is the one result from this
ranking exercise judged solid enough to carry forward into later phases.

### T-112 — Freeze corpus v1.0
`src/freeze.py`, `scripts/t112_freeze.py`

Copies all 12 finished splits per task into the immutable release directory
`data/v1.0/task_{n}/`, records a SHA-256 hash per file plus the exact model/decoding
fingerprints that produced them in a `manifest.json`, and makes the directory read-only.
A deliberate tamper test — modifying one byte of one file and re-verifying — is required
to fail, along with two related tests: a deleted file and an unrecorded extra file both
also have to fail verification, since a release with an untracked extra file is just as
compromised as one with a changed file.

**Status: done — all three tasks frozen and verifying successfully.**

| Task | Splits | Rows |
|---|---|---|
| `task_1` | 12 | 91,144 |
| `task_2` | 12 | 26,152 |
| `task_3` | 12 | 6,384 |

Freezing was deliberately done only *after* τ was calibrated per task (T-110), since
`labse_sim` scores and `translation_drift` flags are baked directly into the frozen
splits — freezing earlier would have sealed a threshold that was still going to change.

**One open item was frozen anyway, with the limitation explicitly recorded rather than
implied away:** T-104's native-speaker hand-check has never actually been done — nobody
on the project reads Bengali, Telugu, or Malayalam, so the 20-sentences-per-direction
check was a project-maintainer spot-check of structure and plausibility, not a
linguistic review. The corpus was frozen with this outstanding, and the datasheet (T-114)
states it as a limitation instead of letting the freeze imply the corpus was fully
verified.

### T-113 — Evaluation-condition matrix
`configs/eval_conditions.json`, `src/conditions.py`

Enumerates every valid combination of "train on this split, evaluate on that split" —
covering all 9 cross-language transfer cells, the four typological quadrants
(Indo-Aryan↔Indo-Aryan, Indo-Aryan→Dravidian, Dravidian→Indo-Aryan,
Dravidian→Dravidian), and the "translationese" conditions (the same language, same
labels, differing only in whether the text is the native version or one of its two
machine-translated versions from a different source language) — and validates that no
split is ever used as both a training source and an evaluation target within one
condition.

**Status: done for all three tasks**, but this task's real contribution was discovering
that the naive version of "no split used as both training and evaluation target" is not
actually sufficient — and would have produced a broken evaluation matrix if left
unfixed.

**Why a split-level check alone fails for tasks 2 and 3.** Because those tasks' three
languages hold the *same items* (the T-102 discovery), a check that only looks at split
names — "don't train on `task_2/H/hin` and evaluate on `task_2/H/ben`" — passes cleanly
even when doing exactly that would evaluate the model on its own training sentences,
just written in a different script. Two layers of protection are enforced together:

1. **Split-level** — no split is used as both a training source and evaluation target.
2. **Item-level** — the actual training items and evaluation items are required to be
   disjoint, enforced by partitioning items using a seeded hash of `item_id` (not by
   position, so it automatically survives any reordering of the underlying files, and
   every language independently agrees on the same partition without needing to
   coordinate). Roughly half the items land on each side (measured 50.1%/48.5% split on
   tasks 2 and 3).

A further design choice: **evaluation prefers native (human-written) text over machine
translation wherever a native version exists.** Evaluating on machine-translated text
would confound "the model failed to transfer" with "the translation pipeline failed" —
exactly the conflation this whole project exists to avoid. Since Malayalam has no native
version anywhere, its evaluation data is unavoidably machine-translated, and the
condition explicitly records this (`eval_provenance`) so the compromise is visible rather
than hidden inside a results table.

Task 1 needs no item-level partition at all, because its blocks already hold genuinely
different content — but its evaluation cells still deliberately cross block boundaries
(e.g. never train and evaluate within block H alone), because block H's Bengali split is
a translation of block H's own Hindi split, and evaluating within one block would repeat
the training-on-your-own-content mistake in a different form.

### T-114 — Datasheet
`src/datasheet.py`, `scripts/t114_datasheet.py`, `data/v1.0/DATASHEET.md`

Generates the corpus's public-facing datasheet — direction matrix, ID scheme,
translation model and decoding configuration, per-direction drift and entity-preservation
rates, storage format, and known limitations — **entirely by reading it back out of the
frozen artifacts** the earlier tasks already produced, rather than by anyone typing
numbers in by hand. A `--check` mode fails if the committed datasheet no longer matches
what the underlying data actually says, specifically so that if the corpus is ever
regenerated, a stale datasheet gets caught rather than quietly going unnoticed (a
document nobody re-reads because it "looks finished" is exactly the kind of document that
drifts silently from the truth).

**Status: done.** Four limitations are stated in the datasheet without softening, because
the project's philosophy is to report what's unfinished plainly rather than imply the
corpus is more validated than it is:

1. No native Malayalam anywhere — Malayalam results always confound transfer with
   translation quality, and Malayalam has no human reference to calibrate against.
2. No native-speaker verification of any translated split — only a maintainer spot-check
   was ever done, and it covered a small fraction of the data (documented above under
   T-104/T-112).
3. The similarity metric used for quality control rewards literal translation over
   natural paraphrasing, so it should be read as a relative ranking tool, not an absolute
   quality score.
4. Tasks 2 and 3 are not independently sourced across languages, which is exactly why the
   item-level partition in `configs/eval_conditions.json` (T-113) is mandatory rather
   than optional.

While building this, a real bug was found: the freezing step (T-112) had written the
*global* similarity threshold into every task's manifest, so task 3's manifest recorded
τ=0.82 even though its actual `translation_drift` flags had been computed using its own
calibrated τ=0.73 — meaning the manifest was misdescribing the very column it was
supposed to certify. This was fixed to record the per-task value, and task 3 was
re-frozen with the corrected manifest.

## Script and digit reference

Used throughout the entity checker (T-105) and the integrity checker (T-107):

| Script | Digits | Codepoints |
|---|---|---|
| Bengali | `০-৯` | U+09E6–U+09EF |
| Devanagari | `०-९` | U+0966–U+096F |
| Telugu | `౦-౯` | U+0C66–U+0C6F |
| Malayalam | `൦-൯` | U+0D66–U+0D6F |
| ASCII | `0-9` | U+0030–U+0039 |

| Script block | Range |
|---|---|
| Devanagari | U+0900–U+097F |
| Bengali | U+0980–U+09FF |
| Telugu | U+0C00–U+0C7F |
| Malayalam | U+0D00–U+0D7F |

One caveat that mattered in practice: **the danda (`।`, U+0964) and double danda
(`॥`, U+0965) are excluded from every script tally**, even though they technically sit
inside the Devanagari Unicode block, because Bengali, Telugu and Malayalam all also use
the danda as ordinary sentence-ending punctuation. Without this exclusion, the
script-leakage check produced 2,211 false positives flagging essentially every Bengali
sentence in task 2 as "containing Devanagari."

## Testing

Every module in `src/` has a corresponding test file in `tests/`. Notable
project-specific fixtures required by the working agreement:

- `entities.py` — at least 5 positive and 5 negative fixtures per check; equivalent
  quantities expressed with different scale words or scripts must compare as equal;
  corrupted rows must never be reported as numeral loss.
- `corpus_io.py` — a byte-for-byte round trip covering all four scripts, currency
  symbols, and lakh/crore expressions.
- `align.py` — IDs are derived from content and stable; conflicting labels across
  languages are caught as an error; the independently-sourced task (task 1) is refused
  rather than silently aligned.
- `integrity.py` — a deliberately script-leaked fixture is caught; the danda is
  confirmed *not* to be flagged as leakage; running the flag-writer twice produces
  identical output to running it once.

## Working agreement for this phase

- Ask before changing the corpus design in the locked section above — it has already
  been decided, reversed, and re-decided once.
- Report row-count anomalies immediately rather than working around them.
- If a translation direction fails, report it and move on — never silently substitute a
  different model or drop the direction.
- When a check fails, show the actual failing rows, not just a count.
- Commit after every self-contained unit of completed work.
