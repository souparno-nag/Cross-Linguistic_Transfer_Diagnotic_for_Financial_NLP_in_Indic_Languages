# Cross-encoder comparison — T-504

Every encoder on identical conditions: the same frozen corpus, the same evaluation matrix (`configs/eval_conditions.json`), the same three seeds, the same metric definitions. Computes nothing new about any single encoder — the per-encoder numbers are T-304's bootstrap over T-306's paired join — and adds only the difference between encoders on the same items.

**Confidence intervals are per seed.** T-304 bootstraps each (condition, seed) independently, so the interval shown is the mean of the three seeds' 1000-sample intervals, not a single interval over pooled seeds. The latter would be narrower and would not be earned.

**Protocol.** Identical across encoders except for one measured, VRAM-forced difference: `task2_hin_mbert` uses `batch_size 8` with `grad_accum 2` where its IndicBERT sibling uses 16 with 1. The effective batch is 16 in both, and because the loss is a mean, accumulating two micro-batches of 8 produces the same gradient as one batch of 16 — the optimiser, learning rate, schedule and update count are unchanged. mBERT could not hold 16 rows of `max_len` 192 on a 3.68 GiB card (T-502, reproduced in a clean process).

**What is missing, and why.** `xlm-r-base` — VRAM-blocked (T-500): 279M parameters need 4.14 GiB of fp32 weights, gradients and AdamW state before any activation, against 3.68 GiB on this card, so it does not fit at batch size 1. Twelve of the eighteen transfer cells per task, and two of the three translationese families, are blocked for **both** encoders because Phase 2 deferred the Bengali and Telugu baselines (CLAUDE2.md); they are shown as blocked rather than dropped, so the matrix keeps its full shape.

**Not interpreted here.** Which encoder wins, and whether a win reflects multilingual capacity dilution or simply parameter count, is T-505's question — mBERT has 179M parameters against IndicBERT-v2's 34M, and this table cannot separate those two explanations.

## In-language baselines

Hindi native split, held-out test fold, three seeds. The two encoders were scored on **identical rows**: `data.stratified_split` is seeded and encoder-blind, so for a given (task, seed) both saw the same test items. Read back from `reports/baselines.parquet`, not transcribed.

| task | encoder | batch × accum | seeds | macro-F1 (mean ± std) |
|---|---|---|---|---|
| 2 | indicbert-v2 | 16 × 1 | 3 | 0.8226 ± 0.0044 |
| 2 | mbert-base | 8 × 2 | 3 | 0.8742 ± 0.0170 |
| 3 | indicbert-v2 | 16 × 1 | 3 | 0.1526 ± 0.0249 |
| 3 | mbert-base | 16 × 1 | 3 | 0.3354 ± 0.0417 |
| — | xlm-r-base | — | 0 | **excluded** — VRAM-blocked (T-500): 279M parameters need 4.14 GiB of fp32 weights, gradients and AdamW state before any activation, against 3.68 GiB on this card, so it does not fit at batch size 1 |

### Task 2 — Transfer — native / cross-block targets

`target` is macro-F1 on the target language; `gap` is source − target on the **same items** (T-306's paired join), so each encoder is measured against itself, not against the other. `Δ target` is the other encoder minus `indicbert-v2`; positive means it scored higher. `beats noise?` asks whether |Δ| exceeds the two encoders' combined seed spread.

| condition | quadrant | indicbert-v2 target | mbert-base target | indicbert-v2 gap [CI] | mbert-base gap [CI] | Δ target | beats noise? |
|---|---|---|---|---|---|---|---|
| transfer_hin_to_ben | Indo-Aryan->Indo-Aryan | 0.5058 ± 0.0819 | 0.8333 ± 0.0182 | 0.4477 ± 0.0716 [0.413, 0.481] | 0.1415 ± 0.0103 [0.118, 0.167] | +0.3275 | yes |
| transfer_hin_to_tel | Indo-Aryan->Dravidian | 0.5009 ± 0.0477 | 0.7383 ± 0.0599 | 0.4526 ± 0.0380 [0.419, 0.487] | 0.2364 ± 0.0555 [0.208, 0.266] | +0.2374 | yes |
| transfer_hin_to_mal | Indo-Aryan->Dravidian | 0.5667 ± 0.0638 | 0.7657 ± 0.0022 | 0.3868 ± 0.0574 [0.352, 0.421] | 0.2090 ± 0.0071 [0.181, 0.238] | +0.1991 | yes |
| transfer_ben_to_hin | Indo-Aryan->Indo-Aryan | blocked | blocked | — | — | — | — |
| transfer_ben_to_tel | Indo-Aryan->Dravidian | blocked | blocked | — | — | — | — |
| transfer_ben_to_mal | Indo-Aryan->Dravidian | blocked | blocked | — | — | — | — |
| transfer_tel_to_hin | Dravidian->Indo-Aryan | blocked | blocked | — | — | — | — |
| transfer_tel_to_ben | Dravidian->Indo-Aryan | blocked | blocked | — | — | — | — |
| transfer_tel_to_mal | Dravidian->Dravidian | blocked | blocked | — | — | — | — |

### Task 2 — Transfer — same-source MT targets

`target` is macro-F1 on the target language; `gap` is source − target on the **same items** (T-306's paired join), so each encoder is measured against itself, not against the other. `Δ target` is the other encoder minus `indicbert-v2`; positive means it scored higher. `beats noise?` asks whether |Δ| exceeds the two encoders' combined seed spread.

| condition | quadrant | indicbert-v2 target | mbert-base target | indicbert-v2 gap [CI] | mbert-base gap [CI] | Δ target | beats noise? |
|---|---|---|---|---|---|---|---|
| transfer_hin_to_ben_mt | Indo-Aryan->Indo-Aryan | 0.4869 ± 0.0990 | 0.8523 ± 0.0117 | 0.4656 ± 0.0883 [0.437, 0.495] | 0.1236 ± 0.0041 [0.103, 0.146] | +0.3654 | yes |
| transfer_hin_to_tel_mt | Indo-Aryan->Dravidian | 0.4881 ± 0.0385 | 0.7509 ± 0.0434 | 0.4645 ± 0.0246 [0.434, 0.494] | 0.2250 ± 0.0404 [0.199, 0.251] | +0.2628 | yes |
| transfer_hin_to_mal_mt | Indo-Aryan->Dravidian | 0.5683 ± 0.0761 | 0.7831 ± 0.0124 | 0.3842 ± 0.0686 [0.356, 0.414] | 0.1928 ± 0.0199 [0.169, 0.217] | +0.2147 | yes |
| transfer_ben_to_hin_mt | Indo-Aryan->Indo-Aryan | blocked | blocked | — | — | — | — |
| transfer_ben_to_tel_mt | Indo-Aryan->Dravidian | blocked | blocked | — | — | — | — |
| transfer_ben_to_mal_mt | Indo-Aryan->Dravidian | blocked | blocked | — | — | — | — |
| transfer_tel_to_hin_mt | Dravidian->Indo-Aryan | blocked | blocked | — | — | — | — |
| transfer_tel_to_ben_mt | Dravidian->Indo-Aryan | blocked | blocked | — | — | — | — |
| transfer_tel_to_mal_mt | Dravidian->Dravidian | blocked | blocked | — | — | — | — |

### Task 3 — Transfer — native / cross-block targets

`target` is macro-F1 on the target language; `gap` is source − target on the **same items** (T-306's paired join), so each encoder is measured against itself, not against the other. `Δ target` is the other encoder minus `indicbert-v2`; positive means it scored higher. `beats noise?` asks whether |Δ| exceeds the two encoders' combined seed spread.

| condition | quadrant | indicbert-v2 target | mbert-base target | indicbert-v2 gap [CI] | mbert-base gap [CI] | Δ target | beats noise? |
|---|---|---|---|---|---|---|---|
| transfer_hin_to_ben | Indo-Aryan->Indo-Aryan | 0.0862 ± 0.0271 | 0.2897 ± 0.0526 | 0.5096 ± 0.1676 [0.449, 0.560] | 0.5155 ± 0.0490 [0.449, 0.579] | +0.2036 | yes |
| transfer_hin_to_tel | Indo-Aryan->Dravidian | 0.0817 ± 0.0398 | 0.2892 ± 0.0674 | 0.5141 ± 0.1701 [0.457, 0.563] | 0.5160 ± 0.0725 [0.450, 0.581] | +0.2075 | yes |
| transfer_hin_to_mal | Indo-Aryan->Dravidian | 0.0923 ± 0.0358 | 0.2441 ± 0.0464 | 0.5035 ± 0.1843 [0.445, 0.554] | 0.5611 ± 0.0499 [0.494, 0.622] | +0.1519 | yes |
| transfer_ben_to_hin | Indo-Aryan->Indo-Aryan | blocked | blocked | — | — | — | — |
| transfer_ben_to_tel | Indo-Aryan->Dravidian | blocked | blocked | — | — | — | — |
| transfer_ben_to_mal | Indo-Aryan->Dravidian | blocked | blocked | — | — | — | — |
| transfer_tel_to_hin | Dravidian->Indo-Aryan | blocked | blocked | — | — | — | — |
| transfer_tel_to_ben | Dravidian->Indo-Aryan | blocked | blocked | — | — | — | — |
| transfer_tel_to_mal | Dravidian->Dravidian | blocked | blocked | — | — | — | — |

### Task 3 — Transfer — same-source MT targets

`target` is macro-F1 on the target language; `gap` is source − target on the **same items** (T-306's paired join), so each encoder is measured against itself, not against the other. `Δ target` is the other encoder minus `indicbert-v2`; positive means it scored higher. `beats noise?` asks whether |Δ| exceeds the two encoders' combined seed spread.

| condition | quadrant | indicbert-v2 target | mbert-base target | indicbert-v2 gap [CI] | mbert-base gap [CI] | Δ target | beats noise? |
|---|---|---|---|---|---|---|---|
| transfer_hin_to_ben_mt | Indo-Aryan->Indo-Aryan | 0.0955 ± 0.0387 | 0.3599 ± 0.0379 | 0.5003 ± 0.1671 [0.441, 0.553] | 0.4453 ± 0.0280 [0.375, 0.516] | +0.2643 | yes |
| transfer_hin_to_tel_mt | Indo-Aryan->Dravidian | 0.0843 ± 0.0358 | 0.3050 ± 0.0821 | 0.5115 ± 0.1742 [0.456, 0.561] | 0.5002 ± 0.0805 [0.435, 0.567] | +0.2207 | yes |
| transfer_hin_to_mal_mt | Indo-Aryan->Dravidian | 0.0803 ± 0.0230 | 0.2776 ± 0.0377 | 0.5155 ± 0.1887 [0.456, 0.567] | 0.5276 ± 0.0355 [0.458, 0.593] | +0.1974 | yes |
| transfer_ben_to_hin_mt | Indo-Aryan->Indo-Aryan | blocked | blocked | — | — | — | — |
| transfer_ben_to_tel_mt | Indo-Aryan->Dravidian | blocked | blocked | — | — | — | — |
| transfer_ben_to_mal_mt | Indo-Aryan->Dravidian | blocked | blocked | — | — | — | — |
| transfer_tel_to_hin_mt | Dravidian->Indo-Aryan | blocked | blocked | — | — | — | — |
| transfer_tel_to_ben_mt | Dravidian->Indo-Aryan | blocked | blocked | — | — | — | — |
| transfer_tel_to_mal_mt | Dravidian->Dravidian | blocked | blocked | — | — | — | — |

### Task 2 — translationese, hin

Same language and labels in all three arms; only provenance differs. `Δ vs native` is each MT arm minus the native arm for that encoder.

| encoder | arm | origin | src_lang | macro-F1 (mean ± std) | Δ vs native |
|---|---|---|---|---|---|
| indicbert-v2 | B | mt | ben | 0.8848 ± 0.0102 | -0.0688 |
| indicbert-v2 | H | native | — | 0.9536 ± 0.0157 | +0.0000 |
| indicbert-v2 | T | mt | tel | 0.8962 ± 0.0189 | -0.0574 |
| mbert-base | B | mt | ben | 0.9443 ± 0.0143 | -0.0329 |
| mbert-base | H | native | — | 0.9772 ± 0.0050 | +0.0000 |
| mbert-base | T | mt | tel | 0.9531 ± 0.0091 | -0.0240 |

### Task 2 — translationese, ben

Same language and labels in all three arms; only provenance differs. `Δ vs native` is each MT arm minus the native arm for that encoder.

| encoder | arm | origin | src_lang | macro-F1 (mean ± std) | Δ vs native |
|---|---|---|---|---|---|
| indicbert-v2 | — | — | — | blocked | — |
| mbert-base | — | — | — | blocked | — |

### Task 2 — translationese, tel

Same language and labels in all three arms; only provenance differs. `Δ vs native` is each MT arm minus the native arm for that encoder.

| encoder | arm | origin | src_lang | macro-F1 (mean ± std) | Δ vs native |
|---|---|---|---|---|---|
| indicbert-v2 | — | — | — | blocked | — |
| mbert-base | — | — | — | blocked | — |

### Task 3 — translationese, hin

Same language and labels in all three arms; only provenance differs. `Δ vs native` is each MT arm minus the native arm for that encoder.

| encoder | arm | origin | src_lang | macro-F1 (mean ± std) | Δ vs native |
|---|---|---|---|---|---|
| indicbert-v2 | B | mt | ben | 0.3324 ± 0.0797 | -0.2617 |
| indicbert-v2 | H | native | — | 0.5941 ± 0.1854 | +0.0000 |
| indicbert-v2 | T | mt | tel | 0.3275 ± 0.0621 | -0.2667 |
| mbert-base | B | mt | ben | 0.6051 ± 0.0563 | -0.1892 |
| mbert-base | H | native | — | 0.7943 ± 0.0155 | +0.0000 |
| mbert-base | T | mt | tel | 0.6144 ± 0.0631 | -0.1799 |

### Task 3 — translationese, ben

Same language and labels in all three arms; only provenance differs. `Δ vs native` is each MT arm minus the native arm for that encoder.

| encoder | arm | origin | src_lang | macro-F1 (mean ± std) | Δ vs native |
|---|---|---|---|---|---|
| indicbert-v2 | — | — | — | blocked | — |
| mbert-base | — | — | — | blocked | — |

### Task 3 — translationese, tel

Same language and labels in all three arms; only provenance differs. `Δ vs native` is each MT arm minus the native arm for that encoder.

| encoder | arm | origin | src_lang | macro-F1 (mean ± std) | Δ vs native |
|---|---|---|---|---|---|
| indicbert-v2 | — | — | — | blocked | — |
| mbert-base | — | — | — | blocked | — |

