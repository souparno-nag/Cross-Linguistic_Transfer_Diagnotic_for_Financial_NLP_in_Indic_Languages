# Datasheet — Indic financial-NLP parallel corpus v1.0

Generated 2026-09-10 from the frozen manifests and the reports that produced them. Every figure below is read back out of those artefacts rather than transcribed, so this document cannot quietly disagree with the data it describes.

**123,680 rows across 36 splits and three tasks.** Each task is a separate corpus sharing one design.

| Task | Content | Rows | Frozen |
|---|---|---|---|
| `task_1` | numerals in financial text (span annotation, no class label) | 91,144 | 2026-09-10 |
| `task_2` | sustainability sentences (binary classification) | 26,152 | 2026-09-10 |
| `task_3` | ESG news headlines (10-class classification) | 6,384 | 2026-09-10 |

## Motivation

The corpus exists to measure **cross-lingual transfer in financial NLP for Indic languages**, and specifically to separate two failures that look identical in an end-to-end score: a model that cannot transfer to a language, and a translation pipeline that damaged the text on the way. Holding the item constant across four languages is what makes that separable.

It is derived from IndicFinNLP (Ghosh et al., LREC-COLING 2024), which ships Hindi, Bengali and Telugu. Malayalam exists here only in translation.

## Composition

### Direction matrix

Three native source splits, each machine-translated into the other three languages: **9 translation directions, 12 splits per task.**

| Block | Native source | Translated into |
|---|---|---|
| `H` | hin | ben, tel, mal |
| `B` | ben | hin, tel, mal |
| `T` | tel | hin, ben, mal |

| Language | Versions | Native exists? |
|---|---|---|
| Hindi | `H_nat`, `Hi←B`, `Hi←T` | yes |
| Bengali | `B_nat`, `Bn←H`, `Bn←T` | yes |
| Telugu | `T_nat`, `Te←H`, `Te←B` | yes |
| Malayalam | `Ml←H`, `Ml←B`, `Ml←T` | **no** |

### ID scheme

The primary key is `(block_id, item_id, lang)`. `item_id` is **derived from content, never from row position** — a positional id silently re-points at a different item if the upstream release is ever reordered. Two consequences worth knowing before joining anything:

- **A task-1 item is a `(sentence, span)` pair, not a sentence.** A sentence containing three numbers appears three times, once per annotated number, so its id includes the span. 6,776 of 10,640 Hindi rows share a sentence with another row.
- **Exact-duplicate rows carry an occurrence suffix** (`…#1`, `…#2`). They are kept rather than deduplicated, and a content-derived id cannot otherwise separate them.

For tasks 2 and 3 `item_id` is global: the same item carries one id in every language, recovered by alignment (task 3 by its `URL` column, task 2 by mutual nearest-neighbour matching required to agree three ways). For task 1 the id is block-local, because its native splits are independently sourced and no cross-language correspondence exists to recover.

### Row schema

| Column | Type | Notes |
|---|---|---|
| `block_id` | str | `H` / `B` / `T` |
| `item_id` | str | content-derived join key |
| `lang` | str | `hin` / `ben` / `tel` / `mal` |
| `origin` | str | `native` / `mt` |
| `src_lang` | str | null on native rows |
| `text` | str | |
| `label`, `label_id` | str, int | classification tasks; from `labels.json` |
| `number_indic`, `number_english`, `start_posn`, `end_posn`, `magnitude`, `span_recovered` | | task 1 only, replacing the label columns |
| `labse_sim` | float | null on native rows |
| `flags` | list[str] | see below |

### Labels

- `task_2`: 2 classes — `sustainable`, `unsustainable`
- `task_3`: 10 classes — `climate change`, `corporate behavior`, `corporate governance`, `environmental opportunities`, `human capital`, `natural capital`, `pollution waste`, `product liability`, `social opportunities`, `stake holder opposition`
- `task_1`: no class label; it marks a numeral inside a sentence.

Labels are carried through translation unchanged and are never invented, remapped or reordered. `label_id` is the index into the list above.

## Collection and preprocessing

### Machine translation

- **Model**: `ai4bharat/indictrans2-indic-indic-1B` (IndicTrans2, direct Indic→Indic — never pivoted through English, which would add a second error source and make drift attribution meaningless).
- **Decoding**: beam 5, max length 256, no sampling, `use_cache=false` — identical across all 9 directions so drift figures are comparable between them.
- **Decoding fingerprint**: `08e5a3dc7f258d7e`. Artefacts are only comparable within one fingerprint.
- **Identical sentences were translated once** and the result shared, so a task-1 sentence carrying three annotated numbers cannot come back as three different sentences.

### Verification

- **Similarity model**: `sentence-transformers/LaBSE`, embeddings L2-normalised at encode time, cosine over **every** MT pair — nothing sampled.
- **τ is calibrated per task, not inherited**, and the reason it differs is text length rather than translation quality:

| Task | τ | Basis |
|---|---|---|
| `task_1` | 0.82 | **inherited, not calibrated** — no human reference exists |
| `task_2` | 0.82 | confirmed against human translations of the same items |
| `task_3` | 0.73 | the value passing 95% of human translations |

## Per-direction results

`below τ` is T-109's drift rate; `entities kept` is T-105's numeral, currency and percentage preservation rate, compared by **value** so that `১০০ মিলিয়ন` → `10 करोड़` counts as preserved rather than as loss.

### `task_1` — numerals in financial text (span annotation, no class label)

| source   | target   | typology               |   median LaBSE | below τ   | entities kept   | corrupted   | numeral re-found   |
|:---------|:---------|:-----------------------|---------------:|:----------|:----------------|:------------|:-------------------|
| hin      | ben      | Indo-Aryan->Indo-Aryan |          0.935 | 6.5%      | 70.9%           | 5.5%        | 84.7%              |
| hin      | tel      | Indo-Aryan->Dravidian  |          0.937 | 4.7%      | 67.5%           | 1.9%        | 87.9%              |
| hin      | mal      | Indo-Aryan->Dravidian  |          0.906 | 10.8%     | 56.9%           | 2.7%        | 82.7%              |
| tel      | hin      | Dravidian->Indo-Aryan  |          0.916 | 15.5%     | 78.7%           | 11.5%       | 83.2%              |
| ben      | hin      | Indo-Aryan->Indo-Aryan |          0.854 | 29.4%     | 58.9%           | 3.0%        | 81.1%              |
| ben      | tel      | Indo-Aryan->Dravidian  |          0.849 | 31.5%     | 57.8%           | 4.2%        | 82.7%              |
| tel      | ben      | Dravidian->Indo-Aryan  |          0.909 | 19.7%     | 76.7%           | 13.7%       | 81.0%              |
| tel      | mal      | Dravidian->Dravidian   |          0.906 | 18.8%     | 70.8%           | 12.7%       | 80.9%              |
| ben      | mal      | Indo-Aryan->Dravidian  |          0.844 | 34.1%     | 49.8%           | 7.0%        | 76.3%              |

### `task_2` — sustainability sentences (binary classification)

| source   | target   | typology               |   median LaBSE | below τ   | entities kept   | corrupted   |
|:---------|:---------|:-----------------------|---------------:|:----------|:----------------|:------------|
| ben      | hin      | Indo-Aryan->Indo-Aryan |          0.931 | 0.8%      | 94.0%           | 0.0%        |
| hin      | tel      | Indo-Aryan->Dravidian  |          0.944 | 0.2%      | 91.6%           | 0.0%        |
| tel      | hin      | Dravidian->Indo-Aryan  |          0.943 | 0.4%      | 90.8%           | 0.0%        |
| hin      | ben      | Indo-Aryan->Indo-Aryan |          0.949 | 0.3%      | 95.7%           | 0.3%        |
| tel      | ben      | Dravidian->Indo-Aryan  |          0.939 | 0.7%      | 92.1%           | 0.2%        |
| tel      | mal      | Dravidian->Dravidian   |          0.924 | 1.6%      | 93.4%           | 0.1%        |
| ben      | tel      | Indo-Aryan->Dravidian  |          0.923 | 2.4%      | 92.0%           | 0.0%        |
| hin      | mal      | Indo-Aryan->Dravidian  |          0.907 | 4.4%      | 89.5%           | 0.1%        |
| ben      | mal      | Indo-Aryan->Dravidian  |          0.902 | 5.4%      | 89.8%           | 0.1%        |

### `task_3` — ESG news headlines (10-class classification)

| source   | target   | typology               |   median LaBSE | below τ   | entities kept   | corrupted   |
|:---------|:---------|:-----------------------|---------------:|:----------|:----------------|:------------|
| ben      | hin      | Indo-Aryan->Indo-Aryan |          0.898 | 0.4%      | 97.0%           | 0.0%        |
| ben      | tel      | Indo-Aryan->Dravidian  |          0.889 | 1.1%      | 92.7%           | 0.0%        |
| tel      | hin      | Dravidian->Indo-Aryan  |          0.89  | 1.1%      | 91.9%           | 0.0%        |
| hin      | tel      | Indo-Aryan->Dravidian  |          0.903 | 0.8%      | 88.3%           | 0.0%        |
| hin      | ben      | Indo-Aryan->Indo-Aryan |          0.908 | 0.4%      | 93.6%           | 0.4%        |
| ben      | mal      | Indo-Aryan->Dravidian  |          0.87  | 2.1%      | 81.0%           | 0.0%        |
| tel      | ben      | Dravidian->Indo-Aryan  |          0.881 | 1.9%      | 91.5%           | 0.2%        |
| tel      | mal      | Dravidian->Dravidian   |          0.861 | 3.8%      | 86.1%           | 0.2%        |
| hin      | mal      | Indo-Aryan->Dravidian  |          0.859 | 4.5%      | 77.4%           | 0.2%        |

## Flags

**No row is ever dropped.** Not for low similarity, not for empty output, not for translation failure. Below-threshold rows are a reported research category, and every problem found is recorded in `flags` with the row kept:

| Flag | Meaning |
|---|---|
| `translation_drift` | LaBSE similarity below this task's τ |
| `entity_loss` | a numeral, currency or percentage did not survive |
| `placeholder_leak` | an unrestored `<ID n>` — the model translated the placeholder itself, destroying the entity it stood for |
| `escape_leak` | a literal `u09bc`-style escape, from a nukta character |
| `script_leakage` | characters from another Indic script, usually an untranslated brand or domain name |
| `truncation_suspect` | output under half this direction's median length ratio |
| `empty_output` | the model returned nothing |
| `span_not_recovered` | task 1: the annotated numeral could not be found |
| `span_scale_shift` | task 1: the quantity survived, rewritten with a scale word |
| `span_ambiguous` | task 1: the value appears more than once; the first was taken |
| `unaligned` | the item has no counterpart in the other languages |

## Uses

The evaluation conditions are enumerated in `configs/eval_conditions.json` and validated: all 9 transfer cells, the four typological quadrants, and the translationese conditions (same language, same labels, differing only in whether the text was written or translated).

**Read that file before designing an experiment.** For tasks 2 and 3 every split holds the same items in every language, so the obvious setup — train on the Hindi split, evaluate on the Malayalam one — tests the model on its own training sentences in translation. The conditions partition items by a seeded hash of `item_id` to prevent exactly that, and prefer native evaluation text so a failure reads as transfer failure rather than translation failure.

## Known limitations

**No native Malayalam.** IndicFinNLP ships Hindi, Bengali and Telugu only, so every Malayalam split is machine translated. Malayalam results therefore confound transfer with translation quality in a way the other three languages do not, and Malayalam has no human reference against which its τ could be calibrated.

**No native-speaker verification of any target split.** The planned check — 20 sentences per direction read by someone who reads Bengali, Telugu and Malayalam — has not been done. Automated checks establish that numerals, labels, structure and script survive; **nothing here establishes that the sentences mean the right thing.** This applies to every direction, Telugu targets included, and is the most significant open item in the release.

**LaBSE similarity rewards literalness, not adequacy.** Machine translations score *higher* than human translations of the same items, in every language pair where both exist — a human translator paraphrases and the metric penalises it. `labse_sim` is therefore usable for ranking directions against each other and poor as an absolute quality score, and τ is a floor for "plausibly a translation of this", not a quality bar.

**Two silent corruption modes survive in the data, flagged but not repaired.** Entity-placeholder leakage and nukta-driven escape leakage both destroy content while leaving fluent-looking text. They are concentrated in task 1 and rare elsewhere.

**Task 1's offsets are not Unicode-normalised, deliberately.** Its `start_posn`/`end_posn` index the raw upstream string; NFC normalisation would break 880 spans while leaving the text looking fine. Do not normalise task 1 text without recomputing offsets.

**Tasks 2 and 3 are not independently sourced across languages.** Their native splits are human translations of one another, which is why alignment was possible and why the item partition above is mandatory. Task 1 is the only task whose languages hold genuinely different content.

## Distribution

### Storage format

| Artefact | Format | Path |
|---|---|---|
| Corpus splits | Parquet | `data/v1.0/task_{n}/{block}/{lang}.parquet` |
| Manifest | JSON | `data/v1.0/task_{n}/manifest.json` |
| Similarity scores | Parquet | `data/verification/task_{n}/labse_scores.parquet` |
| Alignment maps | Parquet | `data/verification/task_{n}/alignment.parquet` |
| Reports | Markdown + Parquet | `reports/task_{n}/` |

**Parquet, never CSV.** Indic scripts and financial numerals break under CSV quoting and delimiter handling. Never pickle.

### Integrity

Each task's `manifest.json` carries a SHA-256 per file, the row count, and the fingerprints that produced it. Verify with `python -m scripts.t112_freeze --task {n} --verify`. The frozen directories are read-only; a new version gets a new directory rather than an edit.

| Task | Splits | Rows | Translation | τ |
|---|---|---|---|---|
| `task_1` | 12 | 91,144 | `08e5a3dc7f258d7e` | 0.82 |
| `task_2` | 12 | 26,152 | `08e5a3dc7f258d7e` | 0.82 |
| `task_3` | 12 | 6,384 | `08e5a3dc7f258d7e` | 0.73 |

### Licence

**CC BY-NC-SA 4.0**, inherited from IndicFinNLP and binding on this corpus as a derivative work. Full terms in `data/base_paper/license.txt`.

- **BY** — attribute IndicFinNLP (Ghosh et al., *IndicFinNLP: Financial Natural Language Processing for Indian Languages*, LREC-COLING 2024) as the source of the native text, and state that this corpus modifies it by machine translation.
- **NC** — no commercial use.
- **SA** — anything derived from this corpus must be distributed under CC BY-NC-SA 4.0 as well. This includes redistributed subsets and, on the share-alike reading, corpora built by translating or transforming it further.

Model licences are separate and additional: IndicTrans2 is released by AI4Bharat under its own terms and its repositories are access-gated; LaBSE under Apache 2.0.

## Maintenance

`data/v1.0/` is immutable. Regenerating anything produces a new version directory, and this datasheet is regenerated with `python -m scripts.t114_datasheet` so its figures follow the data rather than being maintained by hand.
