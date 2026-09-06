# T-102 — Native split audit

Generated 2026-09-06T19:47:01+00:00 · config hash `cd3b4dc6c39aacf7`

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
  "task": 2,
  "tau": 0.82
}
```

## 1. Provenance

All 9 upstream files re-verified against `data/base_paper/manifest.json`: **all match**.

Source: IndicFinNLP (Ghosh et al., LREC-COLING 2024), Kaggle version 2. Licence terms in `data/base_paper/license.txt`.

## 2. Per-split audit (task 2)

| task | language | script | rows | empty_text | empty_label | duplicate_texts | duplicate_conflicting_labels | non_nfc | replacement_char | joiner_rows | leaked_script_rows | leaked_script_chars | rows_with_native_digits | rows_with_ascii_digits | text_len_min | text_len_median | text_len_max | label_counts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | hindi | Deva | 2238 | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 | {} | 0 | 885 | 34 | 148.0 | 663 | {"unsustainable": 1026, "sustainable": 1212} |
| 2 | bengali | Beng | 2228 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | {} | 837 | 19 | 35 | 141.0 | 712 | {"sustainable": 1203, "unsustainable": 1025} |
| 2 | telugu | Telu | 2072 | 0 | 0 | 3 | 0 | 0 | 0 | 0 | 0 | {} | 0 | 819 | 29 | 144.0 | 741 | {"sustainable": 1119, "unsustainable": 953} |

### Class distribution

| language | label | count | share |
|---|---|---|---|
| hindi | sustainable | 1212 | 0.5416 |
| hindi | unsustainable | 1026 | 0.4584 |
| bengali | sustainable | 1203 | 0.5399 |
| bengali | unsustainable | 1025 | 0.4601 |
| telugu | sustainable | 1119 | 0.5401 |
| telugu | unsustainable | 953 | 0.4599 |

### Label vocabulary

Identical across all three languages: **True**  
Union: `sustainable`, `unsustainable`

## 3. Independence

### 3.1 Task 3 is parallel by construction

Row counts {'hindi': 532, 'bengali': 532, 'telugu': 532}, of which 532 URLs appear in all three languages and 532 rows carry the same URL at the same index. Parallel: **True**.

Task 3 is therefore one set of articles translated three ways, not three independent corpora. It cannot supply the native splits, and is retained instead as a human-translated parallel control set for judging MT quality in T-104 and T-110.

### 3.2 Task 2 — LaBSE tests

| pair | test | n | mean | median | p95 | max | above_tau | above_tau_share |
|---|---|---|---|---|---|---|---|---|
| bengali-hindi | B1_row_aligned | 200 | 0.4009 | 0.4039 | 0.5376 | 0.9074 | 2 | 0.01 |
| bengali-hindi | B1_shuffled_control | 200 | 0.3993 | 0.4068 | 0.5365 | 0.645 | 0 | 0.0 |
| bengali-hindi | B2_nearest_neighbour | 200 | 0.8985 | 0.9065 | 0.9584 | 0.9722 | 187 | 0.935 |
| bengali-telugu | B1_row_aligned | 200 | 0.3899 | 0.3769 | 0.5295 | 0.8093 | 0 | 0.0 |
| bengali-telugu | B1_shuffled_control | 200 | 0.3864 | 0.3927 | 0.5162 | 0.6428 | 0 | 0.0 |
| bengali-telugu | B2_nearest_neighbour | 200 | 0.88 | 0.8973 | 0.9524 | 0.9764 | 177 | 0.885 |
| hindi-telugu | B1_row_aligned | 200 | 0.4245 | 0.4249 | 0.5644 | 0.701 | 0 | 0.0 |
| hindi-telugu | B1_shuffled_control | 200 | 0.4222 | 0.4221 | 0.5776 | 0.684 | 0 | 0.0 |
| hindi-telugu | B2_nearest_neighbour | 200 | 0.9101 | 0.9282 | 0.9673 | 0.9734 | 185 | 0.925 |

**B1** pairs rows by index and compares against an index-shuffled control; a large lift over the control means order-preserved parallelism. **B2** takes, for each sampled source sentence, the maximum cosine over the entire target split — it still fires when rows were shuffled or partially dropped, which is the shape these row counts suggest. The §8 random-pair sampling is the B1 control, not a test in its own right.

#### Calibration controls

| control | pair | n | median | mean | above_tau_share |
|---|---|---|---|---|---|
| positive_known_parallel | hindi-bengali | 200 | 0.8761 | 0.8649 | 0.77 |
| negative_same_domain | hindi-bengali | 200 | 0.4845 | 0.4833 | 0.0 |
| negative_same_domain | bengali-hindi | 200 | 0.5288 | 0.5249 | 0.0 |
| under_test | hindi-bengali | 200 | 0.9078 | 0.8932 | 0.905 |

A raw B2 share is not interpretable on its own — every sentence here is narrow-domain ESG text, so a high score could be topical rather than translational. The negative controls pair genuinely different content across languages; the positive control is task 3, proven parallel by its URL column. The finding below is only meaningful because the negative anchors sit far below τ while the split under test sits at or above the positive anchor.

#### Finding

- `bengali-hindi` — **parallel suspected**. 93.5% of sampled sentences have a nearest neighbour at or above τ=0.82; B1 lift over control +0.0015.
- `bengali-telugu` — **parallel suspected**. 88.5% of sampled sentences have a nearest neighbour at or above τ=0.82; B1 lift over control +0.0035.
- `hindi-telugu` — **parallel suspected**. 92.5% of sampled sentences have a nearest neighbour at or above τ=0.82; B1 lift over control +0.0023.

## 4. Result

Checks that failed:

- independence: bengali-hindi looks parallel, not independent
- independence: bengali-telugu looks parallel, not independent
- independence: hindi-telugu looks parallel, not independent
