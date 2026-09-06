# CLAUDE.md — Phase 1: Parallel Corpus Construction (C1)

Scope of this file: **Phase 1 only.** Do not implement training, evaluation, or the
diagnostic pipeline. If a request drifts into Phase 2+, say so and stop.

---

## 1. What Phase 1 delivers

**Three** frozen, machine-translated, semantically parallel corpora across four Indic
languages, screened for translation drift, each with a datasheet.

§2's design is a **template applied once per IndicFinNLP task**, not a single corpus:

| Task | Content | Corpus type |
|---|---|---|
| `task_1` | numerals in financial text | numeral span (schema variant, §6.1) |
| `task_2` | sustainability sentences | binary classification |
| `task_3` | ESG news headlines | 10-class classification |

Everything downstream (zero-shot transfer measurement, failure diagnostics) depends on
this corpus being **row-aligned, label-preserving, and immutable once frozen**. Silent
row loss here corrupts every result in the project and is not recoverable later.

---

## 2. Corpus design — LOCKED, do not modify

> **⚠ T-102 finding: this section's premise holds for task 1 only. Resolved — see
> §2.2 for the per-task consequences.**
>
> §2 assumes three *independently sourced* native splits. For tasks 2 and 3 they are
> not: their Hindi, Bengali and Telugu splits are the same content in shuffled order.
> For task 2:
> 88.5–93.5% of sampled sentences have a nearest cross-language neighbour at or
> above τ=0.82, against 0.0% for same-domain but genuinely different content, and
> the split under test scores *above* the known-parallel positive control (task 3).
> 80% of matched pairs carry identical numeral sets. Task 3 is parallel too, proven
> outright by its URL column.
>
> **Task 1 is the exception and the only independently sourced multilingual data in
> IndicFinNLP**: median nearest cross-language match 0.58–0.63 with 8–15% above τ,
> far closer to the 0.0% negative control than to task 2's 93%. It is not a
> classification task, but it is uncontaminated. See `reports/native_audit.md` §3.
>
> Consequences, for tasks 2 and 3 only: blocks H, B and T do not hold distinct
> content, so their "12 splits" are ~3× redundant and the cross-block contamination
> T-113 tries to rule out is guaranteed rather than merely risked. An alignment step
> is therefore mandatory for those two — without it you cannot tell which items
> overlap between a training language and an evaluation one. Conversely the
> translationese conditions get *stronger*: native Hindi and MT-Hindi-from-Bengali
> can be compared on the same underlying item, which the original design could not
> do. Task 1 is unaffected and needs no alignment step.

Three native source splits from IndicFinNLP. Each is translated into the other three
languages. **9 translation directions, 12 splits total.**

| Block | Native source | Translation targets |
|-------|---------------|---------------------|
| `H`   | Hindi         | Bengali, Malayalam, Telugu |
| `B`   | Bengali       | Hindi, Malayalam, Telugu |
| `T`   | Telugu        | Hindi, Bengali, Malayalam |

Every block is **4-way parallel**: one native split + three MT splits, all sharing
`item_id` and gold label.

| Language  | Versions available            | Native exists? |
|-----------|-------------------------------|----------------|
| Hindi     | `H_nat`, `Hi←B`, `Hi←T`       | yes |
| Bengali   | `B_nat`, `Bn←H`, `Bn←T`       | yes |
| Telugu    | `T_nat`, `Te←H`, `Te←B`       | yes |
| Malayalam | `Ml←H`, `Ml←B`, `Ml←T`        | **no** |

Malayalam having no native version is a known, documented limitation. Do not attempt to
synthesise one. It follows directly from the upstream data: IndicFinNLP ships Hindi,
Bengali and Telugu only (§7.1).

### 2.1 Human translations as a quality ceiling

Tasks 2 and 3 are already parallel across Hindi, Bengali and Telugu, and those existing
versions were produced by people rather than by IndicTrans2. That is a second, free
asset on top of each task's own corpus.

Once a task's splits are aligned, every Indic→Indic direction among those three
languages has a **human reference translation** for the same item. Use it in T-110 as
the quality ceiling: what a human translation of this content scores bounds what an MT
system can reasonably be expected to reach, which is what makes τ defensible rather
than asserted. Where a reference exists, prefer scoring MT output against it directly
over judging similarity to the source.

Malayalam has no reference, and neither does task 1 in any direction. Those fall back
to source-similarity alone, and T-110 must say so rather than implying one standard
was applied throughout.

### 2.2 Per-task instantiation

The block table above is instantiated once per task. What differs is only whether the
native splits are genuinely independent, which decides whether an alignment step is
needed first:

| Task | Natives independent? | Alignment step | Notes |
|---|---|---|---|
| `task_1` | **yes** — median cross-language match 0.58–0.63, 8–15% above τ | none needed | §2 works exactly as originally written. The only contamination-free task. |
| `task_2` | no — 0.91, 93% above τ | **required**, by embedding match | Human-translated but row-shuffled. 1769 of ~2200 align 3-way. |
| `task_3` | no — 0.88, 77% above τ | free — join on `URL` | 532 rows per language, already row-order aligned. |

Where natives are not independent, the same sentence exists in every language, so a
split used for training in one language must not be used for evaluation in another
without saying so. That is what makes the alignment mandatory rather than convenient:
without it you cannot tell which items overlap. T-113 owns the resulting condition
matrix.

### Language codes (FLORES-style, used by IndicTrans2)

```
hin_Deva   ben_Beng   tel_Telu   mal_Mlym
```

---

## 3. Environment

- **Python 3.11** exactly. Assert it at entry point; fail loudly on mismatch.
- **Linux or WSL2.** IndicTransToolkit is not built or tested for Windows.
- `torch>=2.5`, `transformers>=4.51`, `numpy>=2.1` — required by IndicTransToolkit.
- Translation model: `ai4bharat/indictrans2-indic-indic-1B`
  (fallback on 4 GB VRAM: the distilled 320M variant)
- Similarity model: `sentence-transformers/LaBSE`

Currently provisioned: `env/` (gitignored venv, Python 3.11.15) with `kagglehub`,
`pandas`, `openpyxl`, `pyarrow` and `sentence-transformers`. IndicTransToolkit and its
`transformers` pin arrive with T-104. Run everything as a module from the repo root
(`python -m src.…`, `python -m scripts.…`) so the `REPO_ROOT` anchoring in
`src/download_dataset/paths.py` resolves.

**Risk flagged in T-102, affects T-104.** Installing `sentence-transformers` pulled
`transformers` **5.16.1**. §3's floor of `>=4.51` is satisfied, but IndicTrans2 ships
custom modelling code loaded via `trust_remote_code=True` that was written against
`transformers` 4.x, and the 5.x release removed deprecated APIs. Verify the model loads
before building anything on it in T-104; if it does not, pin `transformers<5` and check
that LaBSE still works under the pin.

Hardware: RTX 3060 (12 GB, primary) and RTX 3050 (4 GB, inference/embedding only).
Assume GPU access is intermittent — every GPU job must be **resumable** and must
checkpoint partial output.

---

## 4. Hard rules

These are not preferences. Violating them invalidates the corpus.

1. **Never drop a row.** Not for low similarity, not for empty output, not for
   translation failure. Flag rows; never delete them. Below-threshold pairs are a
   *reported research category*, not garbage.
2. **Never write corpus data as CSV.** Indic scripts and financial numerals break under
   CSV quoting and delimiter handling. Parquet only. See §5.
3. **Never use pickle for corpus data.** Pickle is fine for throwaway intermediate
   caches only, never for anything under `data/`.
4. **Never pivot through English.** Use direct Indic→Indic translation. An English hop
   adds a second error source and makes drift attribution meaningless.
5. **Never modify `data/v1.0/` after freeze.** It is immutable. New versions get a new
   directory.
6. **Never invent, remap, or reorder labels.** Labels come from `labels.json` and are
   carried through translation unchanged.
7. **Determinism.** Every script takes an explicit seed. Two runs of the same config
   produce byte-identical output.
8. **Log the config hash** with every artefact written.
9. **Never Unicode-normalise task 1's text without recomputing its offsets.** Its
   `start_posn`/`end_posn` index the raw upstream string. NFC decomposes precomposed
   nukta letters (`ड़` `ढ़` `ज़` `য়` `ড়`), lengthening the text, and composes Telugu
   vowel signs, shortening it. All 22786 spans are correct as shipped; normalising
   breaks **880** of them while leaving the text looking fine. Normalisation is not
   cleaning here — it is silent, unrecoverable corruption. Tasks 2 and 3 carry no
   offsets and may be normalised freely (task 2 is already fully NFC; task 3 has 5
   rows that are not).

---

## 5. Storage formats

| Artefact | Format | Path |
|---|---|---|
| Upstream source data | `.xlsx` (upstream's own format) | `data/base_paper/raw/task_{n}/{language}.xlsx` |
| Corpus splits | **Parquet** | `data/raw/task_{n}/{block}/{lang}.parquet` |
| Frozen corpus | **Parquet** | `data/v1.0/task_{n}/{block}/{lang}.parquet` |
| Alignment maps | Parquet | `data/verification/task_{n}/alignment.parquet` |
| Embeddings | `.npy` | `data/verification/task_{n}/emb/` |
| Similarity scores | Parquet | `data/verification/task_{n}/labse_scores.parquet` |
| Label schema, configs, manifests | `.json` | `configs/`, `data/base_paper/manifest.json`, `data/v1.0/manifest.json` |

Every derived path carries a `task_{n}` level. Without it the three corpora collide —
`data/raw/H/hin.parquet` is ambiguous across tasks, and silently overwriting one task's
split with another's is exactly the unrecoverable corruption §1 warns about.
| Reports | `.md` + Parquet | `reports/task_{n}/` — `native_audit.{md,parquet}`, `native_independence.parquet`, `native_independence_controls.parquet` |
| Throwaway caches | `.pkl` | `cache/` (gitignored, never released) |
| kagglehub download cache | kagglehub's own | `.cache/kagglehub/` (gitignored, never released) |

The `.xlsx` row is the one exception to rule 2, and it is not ours to change: it is
upstream's release format, committed byte-for-byte with SHA-256s in
`data/base_paper/manifest.json`. Everything **we** derive is Parquet. Read the
spreadsheets once, at the ingest boundary; no module downstream of T-103 touches
`.xlsx`.

All reads and writes go through **`src/corpus_io.py`**. No module calls
`pd.read_parquet` or `to_parquet` directly. This is how format rules stay enforced.
`src/download_dataset/` predates that rule and is exempt — it only moves upstream
files and never writes corpus data.

---

## 6. Canonical row schema

Every corpus Parquet file has exactly these columns:

| Column | Type | Notes |
|---|---|---|
| `block_id` | str | `H` \| `B` \| `T` |
| `item_id` | str | stable within a block; the join key |
| `lang` | str | `hin` \| `ben` \| `tel` \| `mal` |
| `origin` | str | `native` \| `mt` |
| `src_lang` | str | `null` when `origin == native` |
| `text` | str | |
| `label` | str | must be in `labels.json` |
| `label_id` | int | canonical index |
| `labse_sim` | float | `null` for native rows |
| `flags` | list[str] | e.g. `translation_drift`, `entity_loss`, `empty_output` |

`(block_id, item_id, lang)` is the primary key. Rows never move between tasks, so the
task is carried by the path (§5), not by a column.

### 6.1 Numeral-task schema variant (`task_1`)

Task 1 has no class label — it marks a number inside a sentence. It uses the same
columns as above **except** that `label` and `label_id` are replaced by:

| Column | Type | Notes |
|---|---|---|
| `number_indic` | str | the number as written in the source script |
| `number_english` | str | the same value in ASCII digits |
| `start_posn` | int | character offset into `text`, **end-exclusive** |
| `end_posn` | int | `text[start_posn:end_posn] == number_indic` |
| `magnitude` | int | upstream's scale marker |
| `span_recovered` | bool | `null` for native rows; see below |

Upstream's offsets are exact: `indic[start_posn:end_posn]` reproduces `number_indic`
on 500/500 sampled rows in each language.

**Offsets do not survive translation.** A translated sentence has a different length in
a different script, so `start_posn` is meaningless on an MT row and must be *recovered*
by locating the translated numeral in the output, not carried across. Recovery can
fail — the model may drop, reword or mis-transcribe the number. Record the outcome in
`span_recovered` and flag failures; per §4 rule 1 the row is kept either way. That
failure rate is a headline result, not an error: whether financial numerals survive
Indic→Indic translation is the question this project exists to answer.

---

## 7. Repository layout

`✓` marks what exists today. Everything unmarked is created by the tasks in §8.

```
CLAUDE.md            ✓ this file — now tracked (the .gitignore entry was removed)
requirements.txt     ✓ kagglehub, pandas, openpyxl, pyarrow, sentence-transformers
.gitignore           ✓
env/                 ✓ venv, Python 3.11.15 (gitignored)
.cache/kagglehub/    ✓ upstream download cache (gitignored)

configs/               translation_config.json, labels.json, paths.json
src/
  __init__.py        ✓
  download_dataset/  ✓ upstream fetch; complete, no open work
    paths.py         ✓ REPO_ROOT-anchored paths, TASKS, LANGUAGES, raw_path()
    download.py      ✓ python -m src.download_dataset.download [--force]
  unicode_ranges.py  ✓ §9's script/digit tables, dependency-free
  audit.py           ✓ T-102 audit + independence tests
  corpus_io.py          all reads/writes; format enforcement
  ids.py                key construction, join helpers, validation
  translate.py          IndicTrans2 wrapper, resumable batching
  entities.py           numeral/currency preservation checks
  labse_gate.py         embedding + cosine similarity + thresholding
  integrity.py          structural + script-leakage checks
  freeze.py             hashing, manifest, immutability
scripts/
  __init__.py        ✓
  t102_audit.py      ✓ python -m scripts.t102_audit [--independence]
                       (one thin CLI per task, named by task ID)
data/
  base_paper/        ✓ upstream IndicFinNLP, committed; see §7.1
  raw/task_{n}/         per-block MT output
  verification/task_{n}/  alignment maps, scores, embeddings
  v1.0/task_{n}/        frozen release + manifest.json
reports/task_{n}/    ✓ native_audit.{md,parquet}, native_independence.parquet,
                       native_independence_controls.parquet — one dir per task
tests/
cache/                 gitignored
```

`unicode_ranges.py` and `audit.py` are additions to the original module list, made in
T-102. The first exists because §9's codepoint tables are needed by T-102, T-105 and
T-107 alike and three hand-copied copies would drift; the second because §7's
"one thin CLI per task" leaves nowhere for real audit logic to live.

The corpus modules stay **flat in `src/`**. `download_dataset/` is a package
because it owns paths, a CLI, and a manifest writer together; do not take it as a
precedent and nest the corpus modules.

Two path-config surfaces now exist. `src/download_dataset/paths.py` owns
upstream-dataset paths and is already in use; `configs/paths.json` (T-103) owns
corpus paths. Keep them separate rather than merging one into the other — the first
is import-time constants for a finished component, the second is per-run
configuration that gets hashed into artefacts under rule 8.

### 7.1 `data/base_paper/` — the upstream dataset

Materialised by `python -m src.download_dataset.download`, which copies out of the
kagglehub cache into a flat `raw/task_{n}/{language}.xlsx` shape and writes
`manifest.json` (per-file SHA-256) plus a generated `SOURCE.md`. Re-runs are no-ops
when the tree already matches the manifest.

**Never hand-edit this directory** — `--force` deletes and re-materialises `raw/`.
Treat it as immutable input, the same way §4 rule 5 treats `data/v1.0/` as immutable
output.

Source: IndicFinNLP (Ghosh et al., LREC-COLING 2024), Kaggle slug
`sohomghosh/indicfinnlp-financial-nlp-for-indian-languages`, version 2. Licence in
`license.txt`, upstream column docs in `README.md`, both copied verbatim.

Three tasks ship, each in Hindi, Bengali and Telugu — **no Malayalam upstream**,
which is where §2's "Malayalam has no native version" comes from:

| Task | Content | Key columns | Shape |
|---|---|---|---|
| `task_1` | financial text with numeral annotations | `indic`, `number_english`, `number_indic`, `start_posn`, `end_posn`, `magnitude`, `language` | span annotation, **not classification** |
| `task_2` | sustainability sentences | `sentence_indic`, `label` ∈ {sustainable, unsustainable}, `language` | binary classification |
| `task_3` | ESG news titles | `URL`, `news_title_indic`, `ESG_Theme`, `language` | multi-class classification |

**Settled in T-102.** §1 called the corpus "ESG classification" without saying which
task supplies `H_nat`/`B_nat`/`T_nat`. The audit answers it:

- **Task 2 is the native source.** `sentence_indic` is `text`; `label` is `label`, a
  binary `sustainable` / `unsustainable` shared identically by all three languages.
- **Task 3 cannot be.** Its three languages hold 532 rows each with identical URL sets
  in identical row order — it is one set of articles translated three ways, so it fails
  the independence premise of §2 by construction. Retained as the control set in §2.1.
  Each URL appears once per language; there is no internal duplication.
- **Task 1 is not a classification task** (no label column), so it is out as a corpus
  source *for the ESG label*. Its `number_indic` / `number_english` / `start_posn` /
  `end_posn` columns are the natural fixture source for T-105's numeral checker.
  Its spans are end-exclusive — `indic[start_posn:end_posn]` reproduces
  `number_indic` exactly, 500/500 on a sample from each language. Unlike tasks 2 and
  3 it is **not** parallel across languages (see §2), which makes it the only
  contamination-free option in the dataset.

---

## 8. Tasks

Implement in order. Each task is done only when its acceptance criteria pass as an
automated check, not by eyeball.

### T-102 — Audit native splits, confirm independence
`scripts/t102_audit.py`

Audit `H_nat`, `B_nat`, `T_nat` — that is, `data/base_paper/raw/task_{n}/hindi.xlsx`,
`bengali.xlsx` and `telugu.xlsx` for whichever task §7.1's open question settles on:
row counts, class distribution, encoding, duplicates, empty rows, licence provenance.

Provenance is already recorded: cite the SHA-256s from `data/base_paper/manifest.json`
rather than re-deriving them, and confirm they still verify.

Then confirm the three native splits are **not already translations of each other**.

The original wording here — "sample 200 random cross-language pairs, embed with LaBSE,
report the similarity distribution" — does not test anything: random pairs drawn from
two corpora score low whether or not the corpora are parallel. That sampling is kept as
a control, and the finding rests on three tests:

- **B3, structural.** Row counts and per-class proportions. Near-identical class shares
  across supposedly independent corpora are evidence of a shared source. Free, no model.
  Where a join key exists (task 3's `URL`) this alone is decisive.
- **B1, order-aligned.** Cosine of `(src[i], tgt[i])` against an index-shuffled control.
  Catches parallelism that preserved row order — how task 3 is built.
- **B2, nearest-neighbour.** For each of 200 sampled source sentences, the maximum
  cosine over the **entire** target split. This is the real test: it still fires when
  rows were shuffled or partially dropped, which B1 cannot detect. The written finding
  cites B2.

Seeded explicitly (§4 rule 7) with the config hash logged (rule 8).

Because a raw B2 share is not interpretable in a single narrow domain, the run also
emits **calibration controls**: a positive anchor (the control task, proven parallel)
and negative anchors (genuinely different content in the same domain and language
pair). The finding is only valid if the negative anchors sit far below τ.

**Done when:** `reports/native_audit.md` exists with per-language class histograms, a
duplicate/empty-row count, and a written independence finding; `native_audit.parquet`,
`native_independence.parquet` and `native_independence_controls.parquet` carry the
numbers; the script exits non-zero on any failed check.

Run once per task: `python -m scripts.t102_audit --task {n} --independence`. Each task
writes its own report directory; a single shared path would let each run overwrite the
last.

**Status: DONE for all three tasks.**

| Task | Phase A | Independence verdict |
|---|---|---|
| `task_1` | passes | **independent** — median 0.58–0.64, 8.5–15.5% above τ, inside the negative-control band |
| `task_2` | passes | parallel — median 0.88–0.91, 88.5–93.5% above τ |
| `task_3` | passes | parallel — proven by `URL`; embedding controls skipped as degenerate |

Tasks 2 and 3 exit non-zero on the independence check. That is the correct result, not
a failure to fix: it records that their natives are not independent, which §2.2 handles
with an alignment step.

The NFC check is reported, not failed — see rule 9. For task 1 it is replaced by a
span-integrity check, which does fail if any offset stops pointing at its number.

---

### T-103 — ID scheme, label schema, IO layer
`src/ids.py`, `src/corpus_io.py`, `configs/labels.json`

Build the key scheme in §6. Reconcile class names and ordering across all three native
splits into one canonical `labels.json`. Implement the IO layer that owns every
read/write, including the one-time `.xlsx` → Parquet ingest from `data/base_paper/`.

Depends on §7.1's open question being answered — the task choice fixes the label set.

**Done when:**
- A 4-way join on **each** of blocks H, B, T returns exactly N rows with zero nulls.
- The loader raises on any label outside `labels.json` — it does not coerce or skip.
- Round-trip write→read preserves a fixture containing Malayalam text, Telugu digits,
  a currency symbol, and a lakh/crore expression, byte-for-byte.

This task has no code dependencies — its input, `data/base_paper/`, is already on disk.
Start here, once the task choice in §7.1 is settled.

---

### T-104 — IndicTrans2 pipeline + frozen decoding config
`src/translate.py`, `configs/translation_config.json`

Wrap `indictrans2-indic-indic-1B` with `IndicProcessor` pre/post-processing. Freeze
beam size, max length, batch size, and all processor settings — **identical parameters
across all 9 directions**, so drift figures are comparable between them.

Must be resumable: on restart, skip already-translated `item_id`s.

**Done when:** 20 hand-checked sentences translate correctly for each of the 9
directions; config committed; the 320M distilled variant is benchmarked as the 4 GB
fallback and the choice is documented.

---

### T-105 — Numeral and financial-entity preservation checker
`src/entities.py`

Financial numerals are the semantic core of this domain — the whole project is about
whether models handle them across languages. Verify they survive translation.

Check per direction:
- numeral count and values preserved (across digit systems, see §9)
- currency symbols preserved
- percentages preserved
- lakh/crore scale terms preserved

**Done when:** `reports/entity_preservation.parquet` gives a per-direction rate; any
direction below 95% is flagged; affected rows get `entity_loss` in `flags`.

---

### T-106 — Generate all 9 directions
`scripts/t106_generate.py`

Block H → {ben, mal, tel}; Block B → {hin, mal, tel}; Block T → {hin, ben, mal}.
Labels carried forward under strict row alignment.

Roughly 6–7 GPU-hours. Checkpoint after every batch.

**Done when:** row-count parity with the source split for every direction; zero
unaligned rows; zero silently-empty translations (empty output is flagged, not dropped);
output at `data/raw/task_{n}/{block}/{lang}.parquet`.

---

### T-107 — Structural integrity check
`src/integrity.py`

- All three blocks join 4-way on `item_id`
- No truncation (flag suspicious source/target length ratios)
- No label drift between source and target rows
- No encoding corruption
- **Script leakage**: a Malayalam split must not contain Devanagari, etc. See §9.

**Done when:** `reports/integrity.json` is all green, or every failure is enumerated
with row IDs.

---

### T-108 — LaBSE gate over 100% of MT pairs
`src/labse_gate.py`

Embed source and target, cosine similarity per pair, threshold τ = 0.82. All 9
directions, **every pair, nothing sampled**. Cache embeddings as `.npy`. Runs on the
3050.

**Done when:** `data/verification/task_{n}/labse_scores.parquet` has a score for every MT row;
`labse_sim` is populated in the corpus.

---

### T-109 — Per-direction drift report
`scripts/t109_drift.py`

Similarity distribution and below-τ counts, broken down by direction **and** by gold
class.

**Done when:** `reports/drift_by_direction.md` exists; below-τ rows carry
`translation_drift` in `flags` and **remain in the corpus**.

---

### T-110 — Calibrate τ empirically
`scripts/t110_calibrate.py`

τ = 0.82 is inherited, not derived. This is now the project's **only** verification
layer, so it has to be justified rather than asserted.

Plot score distributions per direction. Export 30 pairs either side of τ per target
language for manual inspection. If more than 20% of any direction falls below τ, the
threshold is measuring the wrong thing — report that rather than defending 0.82.

**Done when:** τ is confirmed with evidence or revised with written rationale;
inspection notes committed.

---

### T-111 — Comparative MT-quality ranking
`scripts/t111_ranking.py`

Rank all 9 directions by reliability, combining drift rate and entity preservation.

Questions the output must answer: Is Hi→Ml worse than Bn→Ml? Are Dravidian-source
directions worse than Indo-Aryan-source ones? Is the Indo-Aryan↔Dravidian penalty
symmetric?

**Done when:** `reports/direction_ranking.md` ranks all 9 with supporting numbers.

---

### T-112 — Freeze corpus v1.0
`src/freeze.py`

Copy all 12 splits **per task** into `data/v1.0/task_{n}/`, write `manifest.json` with
per-file SHA-256, make the directory read-only. A task may be frozen independently once
its own checks pass; the manifest covers whatever is present.

**Done when:** the manifest verifies; a tamper test (modify one file, re-verify) fails
as expected.

---

### T-113 — Evaluation-condition matrix
`configs/eval_conditions.json`

Enumerate which splits act as training source, evaluation target, or held out. Must
cover:
- all 9 transfer cells
- the four typological quadrants (Indo-Aryan↔Indo-Aryan, Indo-Aryan→Dravidian,
  Dravidian→Indo-Aryan, Dravidian→Dravidian)
- **translationese conditions**: `H_nat` vs `Hi←B` vs `Hi←T`, and the same for Bengali
  and Telugu — same language, same labels, differing only in provenance

Non-optional. Hindi, Bengali and Telugu each exist in three versions, so cross-block
contamination is a live risk and must be ruled out by construction.

**Done when:** the matrix is explicit and a validator confirms no split appears as both
training source and evaluation target within one condition.

---

### T-114 — Datasheet
`data/v1.0/DATASHEET.md`

Direction matrix; ID scheme; MT model and decoding config; drift rate per direction;
entity-preservation rate per direction; storage format spec; known limitations (no
native Malayalam; Telugu-target splits have no native-speaker verification); CC
BY-NC-SA 4.0 attribution and share-alike obligations.

---

## 9. Script and digit reference

Digit ranges — used by the entity checker and, later, the orthographic diagnostic module:

| Script | Digits | Codepoints |
|---|---|---|
| Bengali | `০-৯` | U+09E6–U+09EF |
| Devanagari | `०-९` | U+0966–U+096F |
| Telugu | `౦-౯` | U+0C66–U+0C6F |
| Malayalam | `൦-൯` | U+0D66–U+0D6F |
| ASCII | `0-9` | U+0030–U+0039 |

Script blocks — used by the leakage check:

| Script | Range |
|---|---|
| Devanagari | U+0900–U+097F |
| Bengali | U+0980–U+09FF |
| Telugu | U+0C00–U+0C7F |
| Malayalam | U+0D00–U+0D7F |

**Exclude U+0964 (danda) and U+0965 (double danda) from every script tally.** They sit
inside the Devanagari block but are shared Indic punctuation: Bengali, Telugu and
Malayalam all use the danda as a sentence terminator. Counting them as Devanagari flags
essentially every Bengali row as script leakage — it produced 2211 false positives on
`task_2/bengali.xlsx` before being fixed. `src/unicode_ranges.py` owns this exclusion;
use that module rather than re-deriving these tables.

---

## 10. Testing

Every module in `src/` ships with tests. Minimum coverage:

- **`ids.py`** — 4-way join returns N rows on all three blocks; missing row raises
- **`corpus_io.py`** — round-trip preserves all four scripts, currency symbols, and
  lakh/crore expressions
- **`entities.py`** — ≥5 positive and ≥5 negative fixtures per check
- **`labse_gate.py`** — known-similar and known-dissimilar pairs land on the expected
  side of τ
- **`integrity.py`** — a deliberately script-leaked fixture is caught

Run tests before any task is marked done.

---

## 11. Working agreement

- **Ask before deviating from the locked design in §2.** The direction matrix was
  decided, reversed, and re-decided. Do not change it on your own initiative.
- **Report row-count anomalies immediately.** Do not work around them.
- **If a translation direction fails, report and continue.** Do not silently substitute
  a different model or drop the direction.
- **Prefer boring code.** This is a research pipeline that must be reproducible in
  October by someone re-reading it cold.
- **When a check fails, show the failing rows.** A count is not a diagnosis.
- **Commit after every meaningful change.** A completed task, a passing check, a fixed
  bug, or any other self-contained unit of work gets its own commit before moving on —
  don't let unrelated changes pile up uncommitted.