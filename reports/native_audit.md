# T-102 — Native split audit

Generated 2026-09-06T19:58:04+00:00 · config hash `0c1979a999042b36`

```json
{
  "batch_size": null,
  "control_task": 3,
  "languages": [
    "hindi",
    "bengali",
    "telugu"
  ],
  "model": null,
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

| task | language | script | rows | empty_text | empty_label | duplicate_texts | duplicate_conflicting_labels | non_nfc | replacement_char | joiner_rows | leaked_script_rows | leaked_script_chars | rows_with_native_digits | rows_with_ascii_digits | text_len_min | text_len_median | text_len_max | label_counts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3 | hindi | Deva | 532 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | 0 | {} | 0 | 190 | 39 | 86.0 | 159 | {"climate change": 92, "pollution waste": 44, "corporate governance": 91, "environmental opportunities": 72, "human capital": 37, "social opportunities": 27, "natural capital": 50, "product liability": 68, "corporate behavior": 30, "stake holder opposition": 21} |
| 3 | bengali | Beng | 532 | 0 | 0 | 0 | 0 | 0 | 0 | 1 | 0 | {} | 99 | 91 | 30 | 87.0 | 178 | {"climate change": 92, "pollution waste": 44, "corporate governance": 91, "environmental opportunities": 72, "human capital": 37, "social opportunities": 27, "natural capital": 50, "product liability": 68, "corporate behavior": 30, "stake holder opposition": 21} |
| 3 | telugu | Telu | 532 | 0 | 0 | 0 | 0 | 0 | 0 | 214 | 0 | {} | 0 | 188 | 9 | 90.0 | 162 | {"climate change": 92, "pollution waste": 44, "corporate governance": 91, "environmental opportunities": 72, "human capital": 37, "social opportunities": 27, "natural capital": 50, "product liability": 68, "corporate behavior": 30, "stake holder opposition": 21} |

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

## 3. Independence

### 3.1 Task 3 is parallel by construction

Row counts {'hindi': 532, 'bengali': 532, 'telugu': 532}, of which 532 URLs appear in all three languages and 532 rows carry the same URL at the same index. Parallel: **True**.

Task 3 is therefore one set of articles translated three ways, not three independent corpora. It cannot supply the native splits, and is retained instead as a human-translated parallel control set for judging MT quality in T-104 and T-110.

### 3.2 Task 2 — LaBSE tests

_Not run. Re-run with `--independence` to produce this section._

## 4. Result

Checks that failed:

- hindi: 5 rows are not NFC
