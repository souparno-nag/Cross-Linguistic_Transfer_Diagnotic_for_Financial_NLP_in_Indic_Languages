# CLAUDE.md — Phase 1: Parallel Corpus Construction (C1)

Scope of this file: **Phase 1 only.** Do not implement training, evaluation, or the
diagnostic pipeline. If a request drifts into Phase 2+, say so and stop.

---

## 1. What Phase 1 delivers

A frozen, machine-translated, semantically parallel ESG classification corpus across
four Indic languages, screened for translation drift, with a datasheet.

Everything downstream (zero-shot transfer measurement, failure diagnostics) depends on
this corpus being **row-aligned, label-preserving, and immutable once frozen**. Silent
row loss here corrupts every result in the project and is not recoverable later.

---

## 2. Corpus design — LOCKED, do not modify

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
synthesise one.

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

Currently provisioned: `env/` (gitignored venv, Python 3.11.15). `requirements.txt`
pins only `kagglehub` so far — it grows as tasks land, and nothing above is installed
yet. Run everything as a module from the repo root (`python -m src.…`) so the
`REPO_ROOT` anchoring in `src/download_dataset/paths.py` resolves.

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

---

## 5. Storage formats

| Artefact | Format | Path |
|---|---|---|
| Upstream source data | `.xlsx` (upstream's own format) | `data/base_paper/raw/task_{n}/{language}.xlsx` |
| Corpus splits | **Parquet** | `data/raw/{block}/{lang}.parquet` |
| Frozen corpus | **Parquet** | `data/v1.0/{block}/{lang}.parquet` |
| Embeddings | `.npy` | `data/verification/emb/` |
| Similarity scores | Parquet | `data/verification/labse_scores.parquet` |
| Label schema, configs, manifests | `.json` | `configs/`, `data/base_paper/manifest.json`, `data/v1.0/manifest.json` |
| Reports | `.md` + Parquet | `reports/` |
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

`(block_id, item_id, lang)` is the primary key.

---

## 7. Repository layout

`✓` marks what exists today. Everything unmarked is created by the tasks in §8.

```
CLAUDE.md            ✓ this file — gitignored, deliberately not committed
requirements.txt     ✓ kagglehub only so far
.gitignore           ✓
env/                 ✓ venv, Python 3.11.15 (gitignored)
.cache/kagglehub/    ✓ upstream download cache (gitignored)

configs/               translation_config.json, labels.json, paths.json
src/
  __init__.py        ✓
  download_dataset/  ✓ upstream fetch; complete, no open work
    paths.py         ✓ REPO_ROOT-anchored paths, TASKS, LANGUAGES, raw_path()
    download.py      ✓ python -m src.download_dataset.download [--force]
  corpus_io.py          all reads/writes; format enforcement
  ids.py                key construction, join helpers, validation
  translate.py          IndicTrans2 wrapper, resumable batching
  entities.py           numeral/currency preservation checks
  labse_gate.py         embedding + cosine similarity + thresholding
  integrity.py          structural + script-leakage checks
  freeze.py             hashing, manifest, immutability
scripts/               one thin CLI per task, named by task ID
data/
  base_paper/        ✓ upstream IndicFinNLP, committed; see §7.1
  raw/                  per-block MT output
  verification/         scores, embeddings
  v1.0/                 frozen release + manifest.json
reports/
tests/
cache/                 gitignored
```

The eight modules above stay **flat in `src/`**. `download_dataset/` is a package
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

**Open question, blocks T-103.** §1 calls the corpus "ESG classification" but does
not say which task supplies `H_nat`/`B_nat`/`T_nat`. Task 2 and task 3 are both
defensible readings and they imply different `labels.json`, different `text` columns,
and different class counts. Task 1 has no label column and cannot be the source for a
classification corpus as specced — though its numeral spans are the natural fixture
source for T-105. **Resolve this before writing `configs/labels.json`**; §4 rule 6
forbids inventing or remapping labels afterwards.

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

Then confirm the three native splits are **not already translations of each other**:
sample 200 random cross-language pairs, embed with LaBSE, report the similarity
distribution. High similarity would mean hidden parallelism, which changes the whole
design.

**Done when:** `reports/native_audit.md` exists with per-language class histograms, a
duplicate/empty-row count, and a written independence finding.

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
output at `data/raw/{block}/{lang}.parquet`.

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

**Done when:** `data/verification/labse_scores.parquet` has a score for every MT row;
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

Copy all 12 splits into `data/v1.0/`, write `manifest.json` with per-file SHA-256, make
the directory read-only.

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