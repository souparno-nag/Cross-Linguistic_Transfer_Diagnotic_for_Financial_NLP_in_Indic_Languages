# T-102 — Native split audit

Generated 2026-09-06T20:56:54+00:00 · config hash `46b017add11cfd3a`

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
  "task": 3,
  "tau": 0.82
}
```

## 1. Provenance

All 9 upstream files re-verified against `data/base_paper/manifest.json`: **all match**.

Source: IndicFinNLP (Ghosh et al., LREC-COLING 2024), Kaggle version 2. Licence terms in `data/base_paper/license.txt`.

## 2. Per-split audit (task 2)

| task | language | script | rows | empty_text | empty_label | duplicate_texts | duplicate_conflicting_labels | non_nfc | nfc_changes_length | spans_total | spans_correct | spans_broken_by_nfc | replacement_char | joiner_rows | leaked_script_rows | leaked_script_chars | rows_with_native_digits | rows_with_ascii_digits | text_len_min | text_len_median | text_len_max | label_counts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | hindi | Deva | 532 | 0 | 0 | 0 | 0 | 5 | 5 | 0 | 0 | 0 | 0 | 0 | 0 | {} | 0 | 190 | 39 | 86.0 | 159 | {"climate change": 92, "pollution waste": 44, "corporate governance": 91, "environmental opportunities": 72, "human capital": 37, "social opportunities": 27, "natural capital": 50, "product liability": 68, "corporate behavior": 30, "stake holder opposition": 21} |
| 3 | bengali | Beng | 532 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | {} | 99 | 91 | 30 | 87.0 | 178 | {"climate change": 92, "pollution waste": 44, "corporate governance": 91, "environmental opportunities": 72, "human capital": 37, "social opportunities": 27, "natural capital": 50, "product liability": 68, "corporate behavior": 30, "stake holder opposition": 21} |
| 3 | telugu | Telu | 532 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 214 | 0 | {} | 0 | 188 | 9 | 90.0 | 162 | {"climate change": 92, "pollution waste": 44, "corporate governance": 91, "environmental opportunities": 72, "human capital": 37, "social opportunities": 27, "natural capital": 50, "product liability": 68, "corporate behavior": 30, "stake holder opposition": 21} |

### Class distribution

| language | label | count | share |
|---|---|---|---|
| hindi | climate change | 92 | 0.1729 |
| hindi | corporate behavior | 30 | 0.0564 |
| hindi | corporate governance | 91 | 0.1711 |
| hindi | environmental opportunities | 72 | 0.1353 |
| hindi | human capital | 37 | 0.0695 |
| hindi | natural capital | 50 | 0.094 |
| hindi | pollution waste | 44 | 0.0827 |
| hindi | product liability | 68 | 0.1278 |
| hindi | social opportunities | 27 | 0.0508 |
| hindi | stake holder opposition | 21 | 0.0395 |
| bengali | climate change | 92 | 0.1729 |
| bengali | corporate behavior | 30 | 0.0564 |
| bengali | corporate governance | 91 | 0.1711 |
| bengali | environmental opportunities | 72 | 0.1353 |
| bengali | human capital | 37 | 0.0695 |
| bengali | natural capital | 50 | 0.094 |
| bengali | pollution waste | 44 | 0.0827 |
| bengali | product liability | 68 | 0.1278 |
| bengali | social opportunities | 27 | 0.0508 |
| bengali | stake holder opposition | 21 | 0.0395 |
| telugu | climate change | 92 | 0.1729 |
| telugu | corporate behavior | 30 | 0.0564 |
| telugu | corporate governance | 91 | 0.1711 |
| telugu | environmental opportunities | 72 | 0.1353 |
| telugu | human capital | 37 | 0.0695 |
| telugu | natural capital | 50 | 0.094 |
| telugu | pollution waste | 44 | 0.0827 |
| telugu | product liability | 68 | 0.1278 |
| telugu | social opportunities | 27 | 0.0508 |
| telugu | stake holder opposition | 21 | 0.0395 |

### Label vocabulary

Identical across all three languages: **True**  
Union: `climate change`, `corporate behavior`, `corporate governance`, `environmental opportunities`, `human capital`, `natural capital`, `pollution waste`, `product liability`, `social opportunities`, `stake holder opposition`

### Normalisation

5 rows are not in NFC. This is a property of the source, not damage: the text uses precomposed nukta letters (ड़, ঢ়, য় …) which NFC decomposes, lengthening the string, and Telugu vowel signs which NFC composes, shortening it.

## 3. Independence

### 3.1 Task 3 is parallel by construction

Row counts {'hindi': 532, 'bengali': 532, 'telugu': 532}, of which 532 URLs appear in all three languages and 532 rows carry the same URL at the same index. Parallel: **True**.

Task 3 is therefore one set of articles translated three ways, not three independent corpora. It cannot supply the native splits, and is retained instead as a human-translated parallel control set for judging MT quality in T-104 and T-110.

### 3.2 Task 2 — LaBSE tests

| pair | test | n | mean | median | p95 | max | above_tau | above_tau_share |
|---|---|---|---|---|---|---|---|---|
| bengali-hindi | B1_row_aligned | 200 | 0.8649 | 0.8761 | 0.9453 | 0.9775 | 154 | 0.77 |
| bengali-hindi | B1_shuffled_control | 200 | 0.3156 | 0.3112 | 0.4655 | 0.6581 | 0 | 0.0 |
| bengali-hindi | B2_nearest_neighbour | 200 | 0.8649 | 0.8761 | 0.9453 | 0.9775 | 154 | 0.77 |
| bengali-telugu | B1_row_aligned | 200 | 0.8343 | 0.848 | 0.9249 | 0.9534 | 132 | 0.66 |
| bengali-telugu | B1_shuffled_control | 200 | 0.3007 | 0.3043 | 0.4797 | 0.8482 | 1 | 0.005 |
| bengali-telugu | B2_nearest_neighbour | 200 | 0.8353 | 0.848 | 0.9249 | 0.9534 | 132 | 0.66 |
| hindi-telugu | B1_row_aligned | 200 | 0.8758 | 0.8842 | 0.9475 | 0.965 | 174 | 0.87 |
| hindi-telugu | B1_shuffled_control | 200 | 0.3417 | 0.3362 | 0.5099 | 0.9443 | 2 | 0.01 |
| hindi-telugu | B2_nearest_neighbour | 200 | 0.8758 | 0.8842 | 0.9475 | 0.965 | 174 | 0.87 |

**B1** pairs rows by index and compares against an index-shuffled control; a large lift over the control means order-preserved parallelism. **B2** takes, for each sampled source sentence, the maximum cosine over the entire target split — it still fires when rows were shuffled or partially dropped, which is the shape these row counts suggest. The §8 random-pair sampling is the B1 control, not a test in its own right.

#### Calibration controls

_Not run: this **is** task 3, whose parallelism is already proven outright by its URL column (§3.1). Embedding controls would compare it against itself._

A raw B2 share is not interpretable on its own — every sentence here is narrow-domain ESG text, so a high score could be topical rather than translational. The negative controls pair genuinely different content across languages; the positive control is task 3, proven parallel by its URL column. The finding below is only meaningful because the negative anchors sit far below τ while the split under test sits at or above the positive anchor.

#### Finding

- `bengali-hindi` — **parallel suspected**. 77.0% of sampled sentences have a nearest neighbour at or above τ=0.82; B1 lift over control +0.5493.
- `bengali-telugu` — **parallel suspected**. 66.0% of sampled sentences have a nearest neighbour at or above τ=0.82; B1 lift over control +0.5336.
- `hindi-telugu` — **parallel suspected**. 87.0% of sampled sentences have a nearest neighbour at or above τ=0.82; B1 lift over control +0.5341.

## 4. Result

Checks that failed:

- independence: bengali-hindi looks parallel, not independent
- independence: bengali-telugu looks parallel, not independent
- independence: hindi-telugu looks parallel, not independent
