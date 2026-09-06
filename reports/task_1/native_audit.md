# T-102 — Native split audit

Generated 2026-09-06T20:02:09+00:00 · config hash `b16f7deb3658ee05`

```json
{
  "batch_size": 64,
  "control_task": 3,
  "languages": [
    "hindi",
    "bengali",
    "telugu"
  ],
  "model": "sentence-transformers/LaBSE",
  "sample": 200,
  "seed": 20260903,
  "task": 1,
  "tau": 0.82
}
```

## 1. Provenance

All 9 upstream files re-verified against `data/base_paper/manifest.json`: **all match**.

Source: IndicFinNLP (Ghosh et al., LREC-COLING 2024), Kaggle version 2. Licence terms in `data/base_paper/license.txt`.

## 2. Per-split audit (task 2)

| task | language | script | rows | empty_text | empty_label | duplicate_texts | duplicate_conflicting_labels | non_nfc | nfc_changes_length | spans_total | spans_correct | spans_broken_by_nfc | replacement_char | joiner_rows | leaked_script_rows | leaked_script_chars | rows_with_native_digits | rows_with_ascii_digits | text_len_min | text_len_median | text_len_max | label_counts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | hindi | Deva | 10640 | 0 | 0 | 4283 | 0 | 672 | 669 | 10640 | 10640 | 270 | 0 | 1118 | 0 | {} | 68 | 10640 | 23 | 135.0 | 1621 | {} |
| 1 | bengali | Beng | 6130 | 0 | 0 | 2718 | 0 | 876 | 876 | 6130 | 6130 | 606 | 0 | 80 | 0 | {} | 6130 | 555 | 20 | 145.0 | 805 | {} |
| 1 | telugu | Telu | 6016 | 0 | 0 | 2318 | 0 | 12 | 12 | 6016 | 6016 | 4 | 0 | 1287 | 0 | {} | 45 | 6016 | 18 | 117.0 | 2510 | {} |

### Class distribution

|  |
||


### Label vocabulary

Identical across all three languages: **True**  
Union: 

### Normalisation

1560 rows are not in NFC. This is a property of the source, not damage: the text uses precomposed nukta letters (ड़, ঢ়, য় …) which NFC decomposes, lengthening the string, and Telugu vowel signs which NFC composes, shortening it.

**Do not normalise this task.** Its numeral offsets index the raw text, and every one of them is correct as shipped (22786/22786). Applying NFC shifts characters and would break **880** spans while leaving the text looking perfectly fine — silent, unrecoverable corruption of the kind §1 warns about. Normalise only if the offsets are recomputed in the same step.

## 3. Independence

### 3.1 Task 3 is parallel by construction

Row counts {'hindi': 532, 'bengali': 532, 'telugu': 532}, of which 532 URLs appear in all three languages and 532 rows carry the same URL at the same index. Parallel: **True**.

Task 3 is therefore one set of articles translated three ways, not three independent corpora. It cannot supply the native splits, and is retained instead as a human-translated parallel control set for judging MT quality in T-104 and T-110.

### 3.2 Task 2 — LaBSE tests

| pair | test | n | mean | median | p95 | max | above_tau | above_tau_share |
|---|---|---|---|---|---|---|---|---|
| bengali-hindi | B1_row_aligned | 200 | 0.3046 | 0.3068 | 0.4414 | 0.575 | 0 | 0.0 |
| bengali-hindi | B1_shuffled_control | 200 | 0.2963 | 0.2925 | 0.4542 | 0.5765 | 0 | 0.0 |
| bengali-hindi | B2_nearest_neighbour | 200 | 0.6157 | 0.5786 | 0.8928 | 0.9432 | 25 | 0.125 |
| bengali-telugu | B1_row_aligned | 200 | 0.2819 | 0.287 | 0.4182 | 0.5293 | 0 | 0.0 |
| bengali-telugu | B1_shuffled_control | 200 | 0.2707 | 0.2683 | 0.443 | 0.6102 | 0 | 0.0 |
| bengali-telugu | B2_nearest_neighbour | 200 | 0.6162 | 0.5719 | 0.8766 | 0.9532 | 31 | 0.155 |
| hindi-telugu | B1_row_aligned | 200 | 0.3379 | 0.3399 | 0.4923 | 0.6018 | 0 | 0.0 |
| hindi-telugu | B1_shuffled_control | 200 | 0.3192 | 0.3251 | 0.4912 | 0.6071 | 0 | 0.0 |
| hindi-telugu | B2_nearest_neighbour | 200 | 0.6642 | 0.6391 | 0.9152 | 0.98 | 23 | 0.115 |

**B1** pairs rows by index and compares against an index-shuffled control; a large lift over the control means order-preserved parallelism. **B2** takes, for each sampled source sentence, the maximum cosine over the entire target split — it still fires when rows were shuffled or partially dropped, which is the shape these row counts suggest. The §8 random-pair sampling is the B1 control, not a test in its own right.

#### Calibration controls

| control | pair | n | median | mean | above_tau_share |
|---|---|---|---|---|---|
| positive_known_parallel | hindi-bengali | 200 | 0.8761 | 0.8649 | 0.77 |
| negative_same_domain | hindi-bengali | 200 | 0.435 | 0.4433 | 0.005 |
| negative_same_domain | bengali-hindi | 200 | 0.5515 | 0.5857 | 0.1 |
| under_test | hindi-bengali | 200 | 0.59 | 0.6151 | 0.085 |

A raw B2 share is not interpretable on its own — every sentence here is narrow-domain ESG text, so a high score could be topical rather than translational. The negative controls pair genuinely different content across languages; the positive control is task 3, proven parallel by its URL column. The finding below is only meaningful because the negative anchors sit far below τ while the split under test sits at or above the positive anchor.

#### Finding

- `bengali-hindi` — independent. 12.5% of sampled sentences have a nearest neighbour at or above τ=0.82; B1 lift over control +0.0083.
- `bengali-telugu` — independent. 15.5% of sampled sentences have a nearest neighbour at or above τ=0.82; B1 lift over control +0.0112.
- `hindi-telugu` — independent. 11.5% of sampled sentences have a nearest neighbour at or above τ=0.82; B1 lift over control +0.0187.

## 4. Result

All Phase A checks passed.
