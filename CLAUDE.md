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
- Translation model: **`ai4bharat/indictrans2-indic-indic-dist-320M`** — chosen in
  T-104, see §3.5. The 1B remains available via `--fallback` but is not the default.
- Similarity model: `sentence-transformers/LaBSE`

Currently provisioned: `env/` (gitignored venv, Python 3.11.15) with `kagglehub`,
`pandas`, `openpyxl`, `pyarrow` and `sentence-transformers`. IndicTransToolkit and its
`transformers` pin arrive with T-104. Run everything as a module from the repo root
(`python -m src.…`, `python -m scripts.…`) so the `REPO_ROOT` anchoring in
`src/download_dataset/paths.py` resolves.

### 3.1 IndicTrans2 access — gated, do this before T-104

The `ai4bharat/indictrans2-*` repos are **gated** (`gated=auto`). Downloads fail with
`401 … gated repo` until the licence is accepted and the machine is authenticated.
Approval is automatic, so there is no wait for a human reviewer:

1. Accept the terms at <https://huggingface.co/ai4bharat/indictrans2-indic-indic-1B>
   (and the 320M distilled repo if the fallback will be used).
2. Create a read token at <https://huggingface.co/settings/tokens>.
3. `hf auth login` — or export `HF_TOKEN`.

Authenticating also raises the Hub rate limit. Unauthenticated pulls stalled repeatedly
while fetching LaBSE, twice hanging at exactly 1.024 GB with the process alive and
throughput at zero; `curl -C -` with `--speed-limit`/`--speed-time` was needed to finish
it. Do not read a stalled download as a broken environment.

**Download sizes are half what the repo totals suggest.** Each repo ships the weights
twice, as `pytorch_model.bin` *and* `model.safetensors`. Only one is needed: 4.8 GB for
`indic-indic-1B`, 1.3 GB for `indic-indic-dist-320M`.

### 3.2 `transformers` must be pinned below 5 — resolved

**Pinned: `transformers==4.57.6`, `sentence-transformers==5.7.0`.** Do not upgrade
either without re-reading this section.

`sentence-transformers` 6.x requires `transformers>=5`, and installing it pulled 5.16.1.
IndicTrans2 does not work on 5.x. Its released checkpoints carry their own modelling
code loaded with `trust_remote_code=True`, written against 4.x, and 5.x breaks it in
three places:

1. `configuration_indictrans.py` imports `transformers.onnx`, removed in 5.x.
2. `IndicTransToolkit`'s `IndicDataCollator` imports `PreTrainedTokenizerBase` from
   `transformers.tokenization_utils`, moved in 5.x — and `__init__.py` imports every
   submodule eagerly, so the whole package fails.
3. `tokenization_indictrans.py` assigns `self.unk_token` **before** calling
   `super().__init__()`. 5.x's attribute machinery raises
   `AttributeError: IndicTransTokenizer has no attribute _special_tokens_map`.

The first two can be shimmed. The third cannot without patching private internals of a
class loaded from remote code, which would not be reproducible. Pinning is the correct
fix, not a workaround — a shim stack three deep against library internals is exactly
what §11's "reproducible in October by someone re-reading it cold" rules out.

`sentence-transformers` 5.7.0 is the newest release that accepts `transformers` 4.x, and
resolves cleanly. **LaBSE output is unaffected**: after the downgrade, T-102b's task-2
alignment reproduced bit-identically (same 5307 rows, same item ids, frame-equal) and
T-102's independence figures were unchanged to four decimals. §4 rule 7 holds across the
pin.

`sentencepiece` is also required — IndicTrans2's tokenizer needs it and it is not pulled
in automatically.

**`use_cache=False` is mandatory even on 4.x.** IndicTrans2's `modeling_indictrans.py`
reads `past_key_values[0][0].shape[2] if past_key_values is not None else 0`. Modern
`transformers` passes a `Cache` object rather than the legacy tuple: it is not `None`,
so the guard passes, but it is empty on the first decode step, so `[0][0]` is `None` and
generation dies with `AttributeError: 'NoneType' object has no attribute 'shape'`. The
setting lives in `configs/translation_config.json` and costs decode speed; it is a
correctness requirement, not a tuning knob.

### 3.3 Hardware — corrected in T-104

The machine has **one GPU: an RTX 3050 Laptop, 4 GB.** §3's earlier claim of a 12 GB
RTX 3060 as primary was wrong; `torch.cuda.device_count()` is 1. Everything must fit in
4 GB, and `cuda:0` *is* the small card.

The 1B model nonetheless fits: 2.42 GB loaded in fp16, peaking under 2.8 GB at beam 5,
batch 8. The 320M distilled variant is therefore not required on VRAM grounds. What is
required is `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` — without it beam search
OOMs on *fragmentation* rather than genuine exhaustion. `src/translate.py` sets it at
module scope, before torch is first imported.

Measured on short text at beam 5, batch 8: ~3 sentences/s, so the full 92,760-row
generation is roughly 8 hours. Longer task-1 and task-2 sentences will be slower.

GPU access is intermittent — every GPU job must be **resumable** and must checkpoint
partial output.

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
| Embeddings | `.npy` | `data/verification/task_{n}/emb/` (gitignored — regenerable, ~20 MB/task) |
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

`item_id` is derived from content, never from row position — a positional id silently
re-points at a different item if upstream reorders anything. Two consequences:

- **A task-1 item is a `(sentence, span)` pair, not a sentence.** A sentence containing
  three numbers appears three times, once per number; 6776 of 10640 Hindi rows share a
  sentence with another row. Its id therefore includes the span.
- **Exact-duplicate rows get an occurrence suffix** (`…#1`, `…#2`). Task 2 has four such
  rows, identical in both text and label. Rule 1 forbids dropping them, and a
  content-derived id cannot otherwise separate them.

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

configs/
  labels.json        ✓ canonical labels per task; task_1 is null (no labels)
  translation_config.json ✓ frozen decoding, identical for all 9 directions
                       paths.json still to come
src/
  __init__.py        ✓
  download_dataset/  ✓ upstream fetch; complete, no open work
    paths.py         ✓ REPO_ROOT-anchored paths, TASKS, LANGUAGES, raw_path()
    download.py      ✓ python -m src.download_dataset.download [--force]
  unicode_ranges.py  ✓ §9's script/digit tables, dependency-free
  audit.py           ✓ T-102 audit + independence tests
  align.py           ✓ T-102b item correspondence recovery
  corpus_io.py       ✓ all reads/writes; format, schema and label enforcement
  ids.py             ✓ key construction, block mapping, join helpers
  translate.py       ✓ IndicTrans2 wrapper, resumable batching
  entities.py        ✓ scale-aware numeral/currency checks; corruption detection
  labse_gate.py         embedding + cosine similarity + thresholding
  integrity.py          structural + script-leakage checks
  freeze.py             hashing, manifest, immutability
scripts/
  __init__.py        ✓
  t102_audit.py      ✓ python -m scripts.t102_audit --task {n} [--independence]
  t102b_align.py     ✓ python -m scripts.t102b_align --task {n}
  t103_ingest.py     ✓ python -m scripts.t103_ingest --task {n}
  t104_smoke.py      ✓ python -m scripts.t104_smoke --task {n} [--benchmark]
  t105_entities.py   ✓ python -m scripts.t105_entities --task {n} [--from-smoke]
                       (one thin CLI per task, named by task ID)
data/
  base_paper/        ✓ upstream IndicFinNLP, committed; see §7.1
  raw/task_{n}/      ✓ native splits ingested; MT output still to come
cache/translate/       per-batch checkpoints (gitignored)
  verification/task_3/ ✓ alignment.parquet
  verification/task_{n}/  alignment maps, scores, embeddings
  v1.0/task_{n}/        frozen release + manifest.json
reports/task_{n}/    ✓ native_audit.{md,parquet}, native_independence.parquet,
                       native_independence_controls.parquet — one dir per task
tests/
cache/                 gitignored
```

`unicode_ranges.py`, `audit.py` and `align.py` are additions to the original module list, made in
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

### T-102b — Align parallel native splits
`src/align.py`, `scripts/t102b_align.py`

Tasks 2 and 3 hold the same content in every language, but nothing in the released
files says which item is which. Recover that mapping. Without it there is no way to
tell which items overlap between a language used for training and one used for
evaluation, which is what makes the transfer numbers meaningful (§2.2).

`item_id` is derived from the join key, never from row position — a positional id
silently re-points at a different item if upstream reorders anything (§4 rule 7).

- **Task 3** — exact join on `URL`. No model needed. **Done: 532/532 items align
  across all three languages, zero unmatched, and every aligned item carries the same
  label in all three.** `data/verification/task_3/alignment.parquet`.
- **Task 2** — no join key, so alignment is by **mutual** nearest neighbour on LaBSE
  above τ, required to be **three-way consistent**. Mutual matching stops ten source
  sentences collapsing onto one target; three-way agreement means each pair survived
  two independent routes. **Done: 1769 items align three ways** (from 2029/2019/1833
  pairwise), all carrying identical labels. The 469/459/303 rows that did not align
  are kept in their own splits but cannot form a parallel item.
  `data/verification/task_2/alignment.parquet`.
- **Task 1** — independently sourced. **Must not be aligned**; the CLI refuses it.
  Aligning it would invent a correspondence that does not exist.

Unmatched rows are reported, never a failure — the splits are genuinely different
sizes, and §4 rule 1 keeps them. A conflicting label **is** a failure: the corpus
cannot carry one gold label for an item whose languages disagree.

**Done when:** every item either aligns across all three languages or is reported as
unmatched; no aligned item carries conflicting labels; the script exits non-zero
otherwise. **Tasks 2 and 3 both pass; task 1 is refused by design.**

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

**Status: DONE**, with one criterion necessarily deferred.

| Criterion | State |
|---|---|
| Loader refuses unknown labels, and a `label_id` disagreeing with the schema | passing |
| Byte-for-byte round trip across all four scripts, Telugu digits, `₹`, lakh/crore | passing |
| Cross-language join returns exactly N with zero nulls | passing at **3-way**: task 2 joins to 1769, task 3 to 532 |
| 4-way join | **deferred to T-106** — the fourth arm is Malayalam, which does not exist until it is translated |

Native splits are ingested for all three tasks — 22786 / 6538 / 1596 rows, none
dropped. Task 1's languages correctly join to **zero**: it is independently sourced, so
a cross-language join is meaningless and the test asserts it stays that way.

Ingest keeps unaligned rows (§4 rule 1) with a block-local id and an `unaligned` flag,
so they remain available while staying visibly outside the parallel set.

---

### T-104 — IndicTrans2 pipeline + frozen decoding config
`src/translate.py`, `configs/translation_config.json`

Wrap `indictrans2-indic-indic-1B` with `IndicProcessor` pre/post-processing. Freeze
beam size, max length, batch size, and all processor settings — **identical parameters
across all 9 directions**, so drift figures are comparable between them.

Must be resumable: on restart, skip already-translated `item_id`s.

Decoding is frozen in `configs/translation_config.json`: beam 5, max length 256, no
sampling. `decoding_fingerprint()` hashes everything that can change the output text and
is logged with each artefact (rule 8). Batch size is deliberately **excluded** from that
hash — it must not affect the result, and catching it if it ever does is the point.

The model is loaded **once** and reused across directions. Loading per direction means
nine loads of a 4.8 GB model, and on the 4 GB card an OOM on the second — the same
mistake that broke the LaBSE pass in T-102.

Runs checkpoint after every batch under `cache/translate/`, keyed by `item_id`, so an
interrupted run resumes instead of restarting (§3, intermittent GPU). A failed batch is
flagged and the run continues (§11); empty output is flagged, never dropped (rule 1).

**Done when:** 20 hand-checked sentences translate correctly for each of the 9
directions; config committed; the 320M distilled variant is benchmarked as the 4 GB
fallback and the choice is documented.

**Status: NOT done. Automated criteria pass; the hand-check is outstanding.**

| Criterion | State |
|---|---|
| Config committed | done — 320M selected, see §3.5 |
| 320M benchmarked and the choice documented | done |
| 20 sentences per direction translate **correctly** | **outstanding** — needs a reader of Bengali, Telugu and Malayalam |

All 720 rows were checked programmatically and every numeral difference classified by
hand, which is what surfaced §3.4's corruption modes. That establishes numbers survive;
it does not establish the sentences mean the right thing. Until someone reads
`reports/task_*/t104_smoke_*.parquet`, T-104 is not closed.

All 9 directions translate on every task under both models, with no empty output.

| Model | 180 rows, 9 directions | Digits identical |
|---|---|---|
| 1B | 61.2 s | 99.4% |
| 320M distilled | **18.4 s** | **100%** |

### 3.5 Model choice — 320M, decided in T-104

Measured on task 1, 180 rows, the same 9 directions:

| Model | Time | Digits identical | Full T-106 run |
|---|---|---|---|
| `indic-indic-1B` | 105.8 s | 86.1% | ~15 h |
| **`indic-indic-dist-320M`** | **12.7 s** | 82.8% | **~2–5 h** |

**8.3× slower for 3.3 points on a digit proxy**, and on task 3 the 320M was *better*
(100% vs 99.4%). The 1B also runs at the edge of 4 GB: `ben→tel` took 80.7 s against
2–6 s elsewhere, the OOM backoff repeatedly halving batches, so its real cost may exceed
15 h.

The digit proxy says nothing about fluency, where a distilled model usually gives up
more than it does on numbers. That trade was accepted deliberately. Reading the
side-by-side output in `reports/task_*/t104_smoke_*.parquet` is the outstanding check.

Switching the model changes `decoding_fingerprint()`, which is intended — artefacts are
only comparable within one fingerprint. Nothing had been generated under the old one.

**A fixed batch size cannot work at 4 GB; batching must adapt.** Task 1's Telugu split
has 280 rows over 1000 characters against a median of 117, so any batch sized for
typical rows OOMs on the rare long one. Beam search holds `beam × batch × vocab` logits
in float32 and IndicTrans2's vocabulary is large — that allocation fails first.
`translate_adaptive()` halves the batch and retries on OOM, down to a single row, and
re-raises only if one row alone cannot fit. Order is preserved.

**Free the allocator between directions.** Activations are freed but their blocks are
not returned, so each direction starts with less room than the last; the 1B run died
several directions in with ~1 GB held beyond the model's 2.42 GB. `Translator.free()`
is called after every direction.

**All three tasks now translate end to end on the 320M, with no empty output.**

| Task | 180 rows | Digits identical | True numeral loss |
|---|---|---|---|
| `task_1` | 12.7 s | 82.8% | **1.7%** (3 rows) |
| `task_2` | 118.8 s | 97.2% | — |
| `task_3` | 18.4 s | 100% | — |

**Digit-identity is not an accuracy score.** Classifying task 1's 31 differing rows by
hand: 12 are correct scale conversions, 6 are scale expansions, 10 are corrupted output
(below), and only **3 are genuine numeral loss**. Reporting 17% loss where the truth is
1.7% would have made task 1 look like the worst direction set in T-111 when it is not.

### 3.4 Two silent corruption modes in IndicTrans2 output — found in T-104

Each hits ~1% of rows. Both destroy content while leaving fluent-looking text, so
neither is visible without an explicit check. **T-107 must detect both; rows carrying
them must be flagged, not silently kept as clean.**

**Corrected in T-105: the real rate is 3.2%, not ~1%, and it affects every target
language.** The first count used a detector covering only two of the four target
scripts. The escape leak transliterates into whichever script is the target, so
`u09bc` appears as `യു09ബിസി`, `ইউ09`, `यू09बीसी` or `యు09` — matching two of those
undercounted it by more than half.

| | rows | corrupted | rate |
|---|---|---|---|
| task 1, 320M | 180 | 18 | **10.0%** |
| task 1, 1B | 180 | 8 | 4.4% |
| task 2, 320M | 180 | 3 | 1.7% |
| task 3, both | 360 | 0 | 0% |

By target: hin 5%, mal 3%, tel 3%, ben 2% — not the Bengali/Malayalam-only pattern the
first count suggested.

**The 320M corrupts more than twice as often as the 1B on task 1 (10.0% vs 4.4%).**
That is a quality difference the digit proxy in §3.5 could not see, and it argues
against the 320M more strongly than the 3.3-point digit gap did. Revisit §3.5 before
committing to the full T-106 run.

**Entity-placeholder leakage.** `IndicProcessor` substitutes entities for `<ID n>`
placeholders and restores them after decoding. The model sometimes *translates the
placeholder text itself* — `ID` becomes `আই. ডি.` in Bengali, `ഐ. ഡി.` in Malayalam — so
restoration cannot match it and the real content is lost:

```
SRC  पीआईबी हिंदी (@PIBHindi) July 22, 2019
MT   পি. আই. বি হিন্দি (<আই. ডি. 1>) জুলাই 22,2019     ← @PIBHindi destroyed
```

It also swallows numeric ranges: a source reading `6-555 टन` emerges as `<আই. ডি. 1> টন`.

**Escape leakage — caused by nukta characters, Bengali source only.** A nukta
(`ড়` `য়`, U+09BC) in the source can emerge as the literal escape `u09bc`, transliterated
into the target script. This is not a guess: **all 8 affected rows have a nukta in the
source and no non-nukta row leaked**, a 3.4% rate among nukta-bearing sources.

```
SRC  গড় ফলন প্রায় ৮ মে টন/হেক্টর।
MT   ഗോഡ് _ യു09ബിസി വിളവ് ഏകദേശം 8 മെയ് ടൺ/ഹെക്ടർ ആണ്.
```

Same character class as rule 9. Nukta handling is a recurring hazard in this pipeline,
not a one-off.

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

**Compare values, not digit strings.** Two distinct traps, both measured in T-104:

1. **Scale words.** `১০০ মিলিয়ন` (100 million) → `10 करोड़` (10 crore) is the same
   quantity. Normalise both sides to a number with its scale word applied — 100 × 10⁶
   and 10 × 10⁷ are both 10⁸ — before comparing. Also handle expansion: `50 हजार`
   correctly becomes `50,000`.
2. **Thousands separators.** A comma is a separator only when exactly three digits
   follow. `जुलाई 22, 2019` → `জুলাই 22,2019` must parse as 22 and 2019, not as one
   number 22,2019.

Getting either wrong inflates the loss rate. On task 1 the naive check reported 17%
loss where the true figure is 1.7%, which would have made task 1 look like the worst
direction set in T-111 when it is among the better ones.

**Exclude corrupted rows from the numeral rate.** Rows carrying the placeholder or
escape corruption of §3.4 have lost content for a reason unrelated to numeral handling.
Counting them as numeral loss attributes the failure to the wrong cause — they were 10
of task 1's 31 differing rows.

**Done when:** `reports/entity_preservation.parquet` gives a per-direction rate; any
direction below 95% is flagged; affected rows get `entity_loss` in `flags`.

**Status: built and validated against real output; awaiting corpus MT.**
`src/entities.py` and `scripts/t105_entities.py`, 32 tests. Run it with
`--from-smoke` until T-106 exists:

```
python -m scripts.t105_entities --task 1 --from-smoke
```

Validation mattered more than the code. Scored against T-104's real output, the first
version reported 68–94% preservation; inspecting the flagged rows found three bugs in
the *checker*, not the model:

1. `50, 000` — postprocessing emits a space after the comma, so it parsed as 50 and 0
   rather than 50000.
2. The escape leak transliterates into the target script, and only two of four forms
   were matched, so corrupted rows were scored as numeral loss.
3. Currency compared as a literal symbol, so `₹` against `రూ` read as a loss.

After fixing those, `altered` counts fell from 6–8 per direction to 0–1, and the
remaining failures are genuine. **A checker that cries wolf is the failure mode here** —
which is why the fixtures include seven pairs that must compare *equal*.

Current reading on task 1's smoke sample: five directions below 95%, driven by genuine
drops, one currency loss and two percent losses, with 10% of rows excluded as corrupted.

---

### T-106 — Generate all 9 directions
`scripts/t106_generate.py`

Block H → {ben, mal, tel}; Block B → {hin, mal, tel}; Block T → {hin, ben, mal}.
Labels carried forward under strict row alignment. **Run once per task.**

**Cost, corrected.** The original "roughly 6–7 GPU-hours" was for one task of ~2200
rows. Across all three:

| Task | Native rows (H/B/T) | Translations | Rough GPU-hours at ~3k/hr |
|---|---|---|---|
| `task_1` | 10640 / 6130 / 6016 | 68,358 | ~23 |
| `task_2` | 2238 / 2228 / 2072 | 19,614 | ~6.5 |
| `task_3` | 532 / 532 / 532 | 4,788 | ~1.6 |
| **total** | | **92,760** | **~31** |

Task 1 alone is three and a half times task 2. Given intermittent GPU access (§3), run
tasks in ascending cost — task 3 first, so the whole path is proven end to end in under
two hours before committing a day to task 1. Checkpointing after every batch (T-104)
is what makes this survivable; a run that has to restart from zero is not.

**Task 1 needs span recovery, and cannot simply carry its annotations across.** Its
`start_posn`/`end_posn` index the source string; the translation has a different length
in a different script. Re-locate the numeral in the output and record the outcome in
`span_recovered` (§6.1). Failures are flagged and kept, and the failure rate is a
headline result rather than an error.

**Done when:** row-count parity with the source split for every direction; zero
unaligned rows; zero silently-empty translations (empty output is flagged, not dropped);
`span_recovered` populated for every task-1 MT row; output at
`data/raw/task_{n}/{block}/{lang}.parquet`.

---

### T-107 — Structural integrity check
`src/integrity.py`

- All three blocks join 4-way on `item_id`
- No truncation (flag suspicious source/target length ratios)
- No label drift between source and target rows
- No encoding corruption
- **Script leakage**: a Malayalam split must not contain Devanagari, etc. See §9.
- **Entity-placeholder leakage**: an unrestored `<ID n>` in MT output. See §3.4.
  Concentrated in Bengali and Malayalam targets.
- **Escape leakage**: a literal `u09bc`-style escape in MT output, caused by nukta
  characters in Bengali source. See §3.4.

The last two each hit ~1% of rows and leave fluent-looking text, so nothing catches
them without an explicit check. Flag them; per rule 1 the rows stay.

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
- **`entities.py`** — ≥5 positive and ≥5 negative fixtures per check; equivalent
  quantities across scale words and scripts must compare **equal**; corrupted rows must
  not be reported as numeral loss
- **`labse_gate.py`** — known-similar and known-dissimilar pairs land on the expected
  side of τ
- **`integrity.py`** — a deliberately script-leaked fixture is caught
- **`align.py`** — ids are key-derived and stable; every item appears once per
  language; conflicting labels are caught; an independently-sourced task is refused
- **`ids.py`** — task-1 spans yield distinct ids; duplicates get a suffix and unique
  ids do not; Malayalam has no block

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