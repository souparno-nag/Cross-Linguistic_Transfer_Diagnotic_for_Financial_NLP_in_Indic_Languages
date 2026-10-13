# Phase 7 — native vs MT target text (T-705)

Config hash `bc5b24734953a7a2`. Shares are of failed (item, seed) rows (CLAUDE7.md). A difference is a finding only if the two 95% CIs do not overlap and the direction agrees in every replicate (hard rule 5).

Two labels carry a stated limitation (hard rule 4):

- `morphological_masking`: a masked term does not raise the failure rate (masked fails more often than intact in 20 of 44 cells; reports/morphology_null.md).
- `terminology_gap`: salience divergence does not separate failures from correct pairs (AUC 0.48-0.52; reports/saliency_baseline.md).

**Scope.** `transfer` evaluates on native target text, `transfer_mt` on MT target text, same items and source model. Bengali native vs MT also differ in digit system. Malayalam appears in the tables but has no native text, so it is not compared.

## Cause mix by target and text provenance

| task | encoder | target | kind | failures | fail rate | unattributed | linguistic | data | model | translation_drift | tokenizer_fragmentation | morphological_masking | terminology_gap | provenance_specific | cross_lingual_hard | class_collapse | seed_unstable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | indicbert-v2 | ben | transfer | 1,987 | 30.2% | 15.5% [13.5%–17.4%] | 15.4% [13.1%–17.7%] | 49.1% [46.3%–51.9%] | 20.0% [18.1%–22.0%] | 0.0% | 0.0% | 12.2% | 3.2% | 20.6% | 28.5% | 9.8% | 10.2% |
| 2 | indicbert-v2 | ben | transfer_mt | 2,425 | 37.7% | 15.6% [13.8%–17.4%] | 13.2% [11.3%–15.0%] | 48.3% [46.0%–50.8%] | 22.9% [21.1%–25.0%] | 0.2% | 0.0% | 9.8% | 3.2% | 17.2% | 31.1% | 10.6% | 12.3% |
| 2 | indicbert-v2 | hin | transfer | 1,580 | 23.6% | 14.4% [12.4%–16.4%] | 4.3% [3.1%–5.7%] | 41.2% [38.0%–44.5%] | 40.1% [37.3%–43.1%] | 0.0% | 0.0% | 0.0% | 4.3% | 20.8% | 20.4% | 31.1% | 9.0% |
| 2 | indicbert-v2 | hin | transfer_mt | 1,872 | 29.4% | 14.2% [12.3%–15.9%] | 3.4% [2.4%–4.6%] | 41.0% [38.0%–43.8%] | 41.5% [38.8%–44.1%] | 0.3% | 0.0% | 0.0% | 3.0% | 18.1% | 23.0% | 30.6% | 10.9% |
| 2 | indicbert-v2 | mal | transfer | 3,011 | 30.3% | 7.8% [6.7%–8.8%] | 10.6% [8.9%–12.1%] | 43.3% [41.0%–45.6%] | 38.4% [36.2%–40.7%] | 7.8% | 0.2% | 0.8% | 1.8% | 15.3% | 28.0% | 32.6% | 5.7% |
| 2 | indicbert-v2 | mal | transfer_mt | 3,650 | 37.5% | 9.3% [8.2%–10.5%] | 4.9% [4.0%–5.9%] | 45.8% [43.7%–47.9%] | 40.0% [37.9%–42.1%] | 2.7% | 0.0% | 0.5% | 1.7% | 14.4% | 31.4% | 33.2% | 6.8% |
| 2 | indicbert-v2 | tel | transfer | 2,125 | 34.5% | 0.7% [0.3%–1.2%] | 20.6% [18.0%–23.3%] | 44.3% [41.5%–47.0%] | 34.4% [31.8%–36.8%] | 0.0% | 0.1% | 19.6% | 0.9% | 8.9% | 35.3% | 33.1% | 1.3% |
| 2 | indicbert-v2 | tel | transfer_mt | 2,673 | 40.2% | 0.7% [0.3%–1.2%] | 23.1% [20.5%–25.4%] | 42.5% [40.2%–45.0%] | 33.7% [31.4%–36.1%] | 0.6% | 0.0% | 21.5% | 0.9% | 9.1% | 33.4% | 32.2% | 1.5% |
| 2 | mbert-base | ben | transfer | 946 | 14.4% | 13.8% [11.0%–16.9%] | 12.8% [9.8%–16.0%] | 57.0% [53.0%–60.8%] | 16.4% [13.6%–19.4%] | 0.0% | 0.0% | 12.5% | 0.3% | 31.3% | 25.7% | 0.0% | 16.4% |
| 2 | mbert-base | ben | transfer_mt | 1,063 | 16.5% | 15.6% [12.7%–18.6%] | 11.0% [8.5%–13.7%] | 51.1% [47.6%–54.6%] | 22.3% [19.4%–25.5%] | 0.0% | 0.0% | 9.2% | 1.8% | 19.9% | 31.1% | 0.0% | 22.3% |
| 2 | mbert-base | hin | transfer | 763 | 11.4% | 13.6% [10.5%–16.8%] | 1.4% [0.6%–2.4%] | 59.9% [55.9%–63.4%] | 25.0% [21.5%–29.0%] | 0.0% | 0.0% | 0.0% | 1.4% | 24.5% | 35.4% | 0.0% | 25.0% |
| 2 | mbert-base | hin | transfer_mt | 1,013 | 15.9% | 15.9% [13.0%–18.9%] | 1.4% [0.6%–2.3%] | 57.4% [53.9%–61.2%] | 25.4% [22.0%–28.6%] | 0.4% | 0.0% | 0.0% | 1.0% | 24.6% | 32.8% | 0.0% | 25.4% |
| 2 | mbert-base | mal | transfer | 1,549 | 15.6% | 20.1% [17.2%–22.9%] | 13.4% [11.1%–16.0%] | 45.5% [42.6%–48.4%] | 21.0% [18.7%–23.6%] | 9.9% | 1.4% | 0.6% | 1.4% | 24.1% | 21.4% | 0.0% | 21.0% |
| 2 | mbert-base | mal | transfer_mt | 1,836 | 18.9% | 21.2% [18.7%–23.7%] | 7.4% [5.7%–9.3%] | 46.7% [43.9%–49.4%] | 24.7% [22.4%–27.4%] | 4.7% | 1.1% | 0.4% | 1.3% | 20.2% | 26.5% | 0.0% | 24.7% |
| 2 | mbert-base | tel | transfer | 1,147 | 18.6% | 24.0% [20.7%–27.4%] | 18.4% [15.1%–21.6%] | 43.2% [39.6%–47.0%] | 14.4% [12.2%–16.8%] | 0.0% | 0.4% | 17.3% | 0.6% | 23.2% | 20.1% | 0.0% | 14.4% |
| 2 | mbert-base | tel | transfer_mt | 1,397 | 21.0% | 26.6% [23.3%–29.8%] | 20.3% [17.2%–23.2%] | 36.5% [33.2%–39.7%] | 16.7% [14.4%–19.1%] | 0.9% | 0.2% | 18.3% | 0.8% | 14.6% | 21.9% | 0.0% | 16.7% |
| 3 | indicbert-v2 | ben | transfer | 701 | 42.6% | 14.6% [11.7%–17.2%] | 13.0% [9.6%–16.5%] | 65.0% [61.1%–69.2%] | 7.4% [5.2%–9.7%] | 6.3% | 0.0% | 3.4% | 3.3% | 8.7% | 56.3% | 0.0% | 7.4% |
| 3 | indicbert-v2 | ben | transfer_mt | 710 | 43.2% | 14.9% [12.3%–17.9%] | 6.6% [4.5%–8.8%] | 71.5% [68.2%–74.9%] | 6.9% [4.9%–9.0%] | 1.3% | 0.0% | 1.8% | 3.5% | 10.8% | 60.7% | 0.0% | 6.9% |
| 3 | indicbert-v2 | hin | transfer | 549 | 33.4% | 10.9% [8.2%–14.0%] | 4.4% [2.5%–6.7%] | 70.5% [66.5%–74.5%] | 14.2% [11.2%–17.5%] | 3.8% | 0.0% | 0.0% | 0.5% | 10.6% | 59.9% | 8.6% | 5.6% |
| 3 | indicbert-v2 | hin | transfer_mt | 546 | 33.2% | 10.4% [7.8%–13.1%] | 1.3% [0.2%–2.9%] | 76.0% [72.3%–79.6%] | 12.3% [9.5%–15.3%] | 0.9% | 0.0% | 0.0% | 0.4% | 9.9% | 66.1% | 7.9% | 4.4% |
| 3 | indicbert-v2 | mal | transfer | 992 | 40.2% | 11.0% [9.1%–13.1%] | 14.6% [11.8%–17.6%] | 68.8% [65.8%–71.9%] | 5.6% [4.2%–7.1%] | 13.2% | 0.3% | 0.0% | 1.1% | 9.0% | 59.8% | 0.9% | 4.7% |
| 3 | indicbert-v2 | mal | transfer_mt | 1,018 | 41.3% | 12.4% [10.2%–14.5%] | 5.3% [3.6%–7.2%] | 76.1% [73.3%–79.1%] | 6.2% [4.7%–7.7%] | 3.5% | 0.0% | 0.0% | 1.8% | 12.3% | 63.9% | 1.3% | 4.9% |
| 3 | indicbert-v2 | tel | transfer | 788 | 47.9% | 10.2% [7.8%–12.6%] | 12.2% [9.1%–15.5%] | 73.0% [69.4%–76.8%] | 4.7% [3.2%–6.3%] | 5.8% | 0.0% | 5.2% | 1.1% | 7.2% | 65.7% | 1.8% | 2.9% |
| 3 | indicbert-v2 | tel | transfer_mt | 798 | 48.5% | 9.9% [7.7%–12.1%] | 9.5% [7.0%–12.5%] | 76.4% [73.0%–79.8%] | 4.1% [2.8%–5.6%] | 0.6% | 0.0% | 7.6% | 1.3% | 8.9% | 67.5% | 0.0% | 4.1% |
| 3 | mbert-base | ben | transfer | 802 | 48.8% | 11.1% [8.4%–13.9%] | 13.6% [10.3%–17.1%] | 64.8% [61.0%–68.9%] | 10.5% [8.1%–12.9%] | 8.7% | 0.2% | 2.9% | 1.7% | 19.1% | 45.8% | 0.0% | 10.5% |
| 3 | mbert-base | ben | transfer_mt | 739 | 45.0% | 14.3% [11.4%–17.7%] | 5.8% [3.7%–8.1%] | 68.3% [64.3%–72.3%] | 11.5% [8.7%–14.1%] | 0.9% | 0.0% | 3.1% | 1.8% | 14.9% | 53.5% | 0.0% | 11.5% |
| 3 | mbert-base | hin | transfer | 768 | 46.7% | 12.2% [9.5%–15.0%] | 6.4% [4.0%–9.2%] | 68.9% [65.0%–72.6%] | 12.5% [10.0%–15.0%] | 5.2% | 0.0% | 0.0% | 1.2% | 16.8% | 52.1% | 0.0% | 12.5% |
| 3 | mbert-base | hin | transfer_mt | 737 | 44.8% | 15.5% [12.4%–18.5%] | 2.8% [1.4%–4.5%] | 69.5% [65.6%–73.2%] | 12.2% [9.6%–15.1%] | 0.9% | 0.0% | 0.0% | 1.9% | 14.1% | 55.4% | 0.0% | 12.2% |
| 3 | mbert-base | mal | transfer | 1,254 | 50.9% | 15.8% [13.3%–18.4%] | 16.6% [13.6%–19.7%] | 58.1% [54.7%–61.4%] | 9.6% [7.9%–11.3%] | 14.5% | 1.4% | 0.0% | 0.7% | 13.5% | 44.6% | 0.0% | 9.6% |
| 3 | mbert-base | mal | transfer_mt | 1,203 | 48.8% | 19.8% [17.0%–22.5%] | 4.7% [3.0%–6.3%] | 63.9% [60.7%–67.1%] | 11.6% [9.7%–13.6%] | 3.1% | 0.5% | 0.0% | 1.1% | 12.1% | 51.9% | 0.0% | 11.6% |
| 3 | mbert-base | tel | transfer | 816 | 49.6% | 11.4% [9.1%–13.6%] | 14.3% [10.8%–18.1%] | 64.2% [60.3%–68.1%] | 10.0% [7.9%–12.4%] | 8.1% | 0.0% | 5.1% | 1.1% | 18.9% | 45.3% | 0.0% | 10.0% |
| 3 | mbert-base | tel | transfer_mt | 784 | 47.7% | 12.0% [9.5%–14.6%] | 8.5% [5.9%–11.5%] | 66.6% [62.4%–70.5%] | 12.9% [10.4%–15.9%] | 0.8% | 0.0% | 6.9% | 0.9% | 17.2% | 49.4% | 0.0% | 12.9% |

## Failures labelled as the translation's doing

`translation_drift` (the MT lost the meaning) plus `provenance_specific` (wrong on this text, right on the same item in the other provenance), per condition kind. Pooled point shares; their CIs are in the tables above.

| task | encoder | target | kind | translation_drift | provenance_specific | together |
|---|---|---|---|---|---|---|
| 2 | indicbert-v2 | ben | transfer | 0.0% | 20.6% | 20.6% |
| 2 | indicbert-v2 | ben | transfer_mt | 0.2% | 17.2% | 17.4% |
| 2 | indicbert-v2 | hin | transfer | 0.0% | 20.8% | 20.8% |
| 2 | indicbert-v2 | hin | transfer_mt | 0.3% | 18.1% | 18.4% |
| 2 | indicbert-v2 | mal | transfer | 7.8% | 15.3% | 23.1% |
| 2 | indicbert-v2 | mal | transfer_mt | 2.7% | 14.4% | 17.1% |
| 2 | indicbert-v2 | tel | transfer | 0.0% | 8.9% | 8.9% |
| 2 | indicbert-v2 | tel | transfer_mt | 0.6% | 9.1% | 9.7% |
| 2 | mbert-base | ben | transfer | 0.0% | 31.3% | 31.3% |
| 2 | mbert-base | ben | transfer_mt | 0.0% | 19.9% | 19.9% |
| 2 | mbert-base | hin | transfer | 0.0% | 24.5% | 24.5% |
| 2 | mbert-base | hin | transfer_mt | 0.4% | 24.6% | 25.0% |
| 2 | mbert-base | mal | transfer | 9.9% | 24.1% | 34.1% |
| 2 | mbert-base | mal | transfer_mt | 4.7% | 20.2% | 24.8% |
| 2 | mbert-base | tel | transfer | 0.0% | 23.2% | 23.2% |
| 2 | mbert-base | tel | transfer_mt | 0.9% | 14.6% | 15.5% |
| 3 | indicbert-v2 | ben | transfer | 6.3% | 8.7% | 15.0% |
| 3 | indicbert-v2 | ben | transfer_mt | 1.3% | 10.8% | 12.1% |
| 3 | indicbert-v2 | hin | transfer | 3.8% | 10.6% | 14.4% |
| 3 | indicbert-v2 | hin | transfer_mt | 0.9% | 9.9% | 10.8% |
| 3 | indicbert-v2 | mal | transfer | 13.2% | 9.0% | 22.2% |
| 3 | indicbert-v2 | mal | transfer_mt | 3.5% | 12.3% | 15.8% |
| 3 | indicbert-v2 | tel | transfer | 5.8% | 7.2% | 13.1% |
| 3 | indicbert-v2 | tel | transfer_mt | 0.6% | 8.9% | 9.5% |
| 3 | mbert-base | ben | transfer | 8.7% | 19.1% | 27.8% |
| 3 | mbert-base | ben | transfer_mt | 0.9% | 14.9% | 15.8% |
| 3 | mbert-base | hin | transfer | 5.2% | 16.8% | 22.0% |
| 3 | mbert-base | hin | transfer_mt | 0.9% | 14.1% | 15.1% |
| 3 | mbert-base | mal | transfer | 14.5% | 13.5% | 28.0% |
| 3 | mbert-base | mal | transfer_mt | 3.1% | 12.1% | 15.1% |
| 3 | mbert-base | tel | transfer | 8.1% | 18.9% | 27.0% |
| 3 | mbert-base | tel | transfer_mt | 0.8% | 17.2% | 18.0% |

## Comparisons

### Task 2: ben target, native vs MT text

A = native ben; B = MT ben. Replicated over encoder: indicbert-v2, mbert-base.

| level | measure | A (indicbert-v2) | B (indicbert-v2) | A (mbert-base) | B (mbert-base) | verdict |
|---|---|---|---|---|---|---|
| L1 | linguistic | 15.4% [13.1%–17.7%] | 13.2% [11.3%–15.0%] | 12.8% [9.8%–16.0%] | 11.0% [8.5%–13.7%] | within noise |
| L1 | data | 49.1% [46.3%–51.9%] | 48.3% [46.0%–50.8%] | 57.0% [53.0%–60.8%] | 51.1% [47.6%–54.6%] | within noise |
| L1 | model | 20.0% [18.1%–22.0%] | 22.9% [21.1%–25.0%] | **16.4% [13.6%–19.4%]** | **22.3% [19.4%–25.5%]** | one encoder only |
| L1 | unattributed | 15.5% [13.5%–17.4%] | 15.6% [13.8%–17.4%] | 13.8% [11.0%–16.9%] | 15.6% [12.7%–18.6%] | within noise |
| L2 | translation_drift | **0.0% [0.0%–0.0%]** | **0.2% [0.0%–0.5%]** | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | one encoder only |
| L2 | tokenizer_fragmentation | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | within noise |
| L2 | morphological_masking | 12.2% [10.0%–14.4%] | 9.8% [8.1%–11.6%] | 12.5% [9.5%–15.8%] | 9.2% [6.8%–11.7%] | within noise |
| L2 | terminology_gap | 3.2% [2.2%–4.2%] | 3.2% [2.4%–4.0%] | **0.3% [0.0%–0.7%]** | **1.8% [0.9%–2.8%]** | one encoder only |
| L2 | provenance_specific | 20.6% [18.5%–22.7%] | 17.2% [15.6%–19.1%] | **31.3% [27.7%–34.9%]** | **19.9% [17.0%–22.9%]** | one encoder only |
| L2 | cross_lingual_hard | 28.5% [26.0%–31.2%] | 31.1% [28.6%–33.6%] | 25.7% [22.0%–29.1%] | 31.1% [27.8%–34.6%] | within noise |
| L2 | class_collapse | 9.8% [8.3%–11.4%] | 10.6% [9.0%–11.9%] | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | within noise |
| L2 | seed_unstable | 10.2% [8.7%–11.9%] | 12.3% [10.8%–13.9%] | **16.4% [13.6%–19.4%]** | **22.3% [19.4%–25.5%]** | one encoder only |

Failures counted — indicbert-v2: A 1,987, B 2,425; mbert-base: A 946, B 1,063. Bold: the two intervals do not overlap in that replicate.

### Task 2: hin target, native vs MT text

A = native hin; B = MT hin. Replicated over encoder: indicbert-v2, mbert-base.

| level | measure | A (indicbert-v2) | B (indicbert-v2) | A (mbert-base) | B (mbert-base) | verdict |
|---|---|---|---|---|---|---|
| L1 | linguistic | 4.3% [3.1%–5.7%] | 3.4% [2.4%–4.6%] | 1.4% [0.6%–2.4%] | 1.4% [0.6%–2.3%] | within noise |
| L1 | data | 41.2% [38.0%–44.5%] | 41.0% [38.0%–43.8%] | 59.9% [55.9%–63.4%] | 57.4% [53.9%–61.2%] | within noise |
| L1 | model | 40.1% [37.3%–43.1%] | 41.5% [38.8%–44.1%] | 25.0% [21.5%–29.0%] | 25.4% [22.0%–28.6%] | within noise |
| L1 | unattributed | 14.4% [12.4%–16.4%] | 14.2% [12.3%–15.9%] | 13.6% [10.5%–16.8%] | 15.9% [13.0%–18.9%] | within noise |
| L2 | translation_drift | **0.0% [0.0%–0.0%]** | **0.3% [0.1%–0.7%]** | **0.0% [0.0%–0.0%]** | **0.4% [0.1%–0.9%]** | held |
| L2 | tokenizer_fragmentation | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | within noise |
| L2 | morphological_masking | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | within noise |
| L2 | terminology_gap | 4.3% [3.1%–5.7%] | 3.0% [2.1%–4.2%] | 1.4% [0.6%–2.4%] | 1.0% [0.3%–1.8%] | within noise |
| L2 | provenance_specific | 20.8% [18.3%–23.1%] | 18.1% [16.1%–19.9%] | 24.5% [21.0%–28.2%] | 24.6% [21.1%–27.8%] | within noise |
| L2 | cross_lingual_hard | 20.4% [17.9%–23.2%] | 23.0% [20.2%–25.4%] | 35.4% [31.3%–39.3%] | 32.8% [29.2%–36.5%] | within noise |
| L2 | class_collapse | 31.1% [28.2%–33.8%] | 30.6% [28.0%–33.3%] | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | within noise |
| L2 | seed_unstable | 9.0% [7.5%–10.7%] | 10.9% [9.3%–12.5%] | 25.0% [21.5%–29.0%] | 25.4% [22.0%–28.6%] | within noise |

Failures counted — indicbert-v2: A 1,580, B 1,872; mbert-base: A 763, B 1,013. Bold: the two intervals do not overlap in that replicate.

### Task 2: tel target, native vs MT text

A = native tel; B = MT tel. Replicated over encoder: indicbert-v2, mbert-base.

| level | measure | A (indicbert-v2) | B (indicbert-v2) | A (mbert-base) | B (mbert-base) | verdict |
|---|---|---|---|---|---|---|
| L1 | linguistic | 20.6% [18.0%–23.3%] | 23.1% [20.5%–25.4%] | 18.4% [15.1%–21.6%] | 20.3% [17.2%–23.2%] | within noise |
| L1 | data | 44.3% [41.5%–47.0%] | 42.5% [40.2%–45.0%] | 43.2% [39.6%–47.0%] | 36.5% [33.2%–39.7%] | within noise |
| L1 | model | 34.4% [31.8%–36.8%] | 33.7% [31.4%–36.1%] | 14.4% [12.2%–16.8%] | 16.7% [14.4%–19.1%] | within noise |
| L1 | unattributed | 0.7% [0.3%–1.2%] | 0.7% [0.3%–1.2%] | 24.0% [20.7%–27.4%] | 26.6% [23.3%–29.8%] | within noise |
| L2 | translation_drift | **0.0% [0.0%–0.0%]** | **0.6% [0.2%–1.1%]** | **0.0% [0.0%–0.0%]** | **0.9% [0.3%–1.7%]** | held |
| L2 | tokenizer_fragmentation | 0.1% [0.0%–0.3%] | 0.0% [0.0%–0.0%] | 0.4% [0.0%–1.2%] | 0.2% [0.0%–0.7%] | within noise |
| L2 | morphological_masking | 19.6% [17.1%–22.3%] | 21.5% [19.0%–23.9%] | 17.3% [14.1%–20.5%] | 18.3% [15.5%–21.2%] | within noise |
| L2 | terminology_gap | 0.9% [0.4%–1.5%] | 0.9% [0.5%–1.5%] | 0.6% [0.1%–1.2%] | 0.8% [0.2%–1.5%] | within noise |
| L2 | provenance_specific | 8.9% [7.6%–10.5%] | 9.1% [7.8%–10.4%] | **23.2% [20.0%–26.5%]** | **14.6% [12.6%–16.8%]** | one encoder only |
| L2 | cross_lingual_hard | 35.3% [32.6%–38.1%] | 33.4% [31.3%–35.8%] | 20.1% [17.2%–23.2%] | 21.9% [19.1%–24.9%] | within noise |
| L2 | class_collapse | 33.1% [30.4%–35.6%] | 32.2% [30.0%–34.5%] | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | within noise |
| L2 | seed_unstable | 1.3% [0.8%–1.9%] | 1.5% [1.0%–1.9%] | 14.4% [12.2%–16.8%] | 16.7% [14.4%–19.1%] | within noise |

Failures counted — indicbert-v2: A 2,125, B 2,673; mbert-base: A 1,147, B 1,397. Bold: the two intervals do not overlap in that replicate.

### Task 3: ben target, native vs MT text

A = native ben; B = MT ben. Replicated over encoder: indicbert-v2, mbert-base.

| level | measure | A (indicbert-v2) | B (indicbert-v2) | A (mbert-base) | B (mbert-base) | verdict |
|---|---|---|---|---|---|---|
| L1 | linguistic | **13.0% [9.6%–16.5%]** | **6.6% [4.5%–8.8%]** | **13.6% [10.3%–17.1%]** | **5.8% [3.7%–8.1%]** | held |
| L1 | data | 65.0% [61.1%–69.2%] | 71.5% [68.2%–74.9%] | 64.8% [61.0%–68.9%] | 68.3% [64.3%–72.3%] | within noise |
| L1 | model | 7.4% [5.2%–9.7%] | 6.9% [4.9%–9.0%] | 10.5% [8.1%–12.9%] | 11.5% [8.7%–14.1%] | within noise |
| L1 | unattributed | 14.6% [11.7%–17.2%] | 14.9% [12.3%–17.9%] | 11.1% [8.4%–13.9%] | 14.3% [11.4%–17.7%] | within noise |
| L2 | translation_drift | **6.3% [4.0%–9.0%]** | **1.3% [0.3%–2.5%]** | **8.7% [6.0%–12.0%]** | **0.9% [0.1%–2.0%]** | held |
| L2 | tokenizer_fragmentation | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | 0.2% [0.0%–0.8%] | 0.0% [0.0%–0.0%] | within noise |
| L2 | morphological_masking | 3.4% [1.6%–5.4%] | 1.8% [0.6%–3.5%] | 2.9% [1.4%–4.6%] | 3.1% [1.4%–5.1%] | within noise |
| L2 | terminology_gap | 3.3% [2.0%–4.8%] | 3.5% [2.2%–5.0%] | 1.7% [0.8%–3.0%] | 1.8% [0.8%–2.9%] | within noise |
| L2 | provenance_specific | 8.7% [6.6%–11.0%] | 10.8% [8.6%–13.2%] | 19.1% [15.9%–22.5%] | 14.9% [12.2%–17.8%] | within noise |
| L2 | cross_lingual_hard | 56.3% [52.6%–60.3%] | 60.7% [57.2%–64.5%] | 45.8% [42.0%–50.1%] | 53.5% [49.2%–57.9%] | within noise |
| L2 | class_collapse | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | within noise |
| L2 | seed_unstable | 7.4% [5.2%–9.7%] | 6.9% [4.9%–9.0%] | 10.5% [8.1%–12.9%] | 11.5% [8.7%–14.1%] | within noise |

Failures counted — indicbert-v2: A 701, B 710; mbert-base: A 802, B 739. Bold: the two intervals do not overlap in that replicate.

### Task 3: hin target, native vs MT text

A = native hin; B = MT hin. Replicated over encoder: indicbert-v2, mbert-base.

| level | measure | A (indicbert-v2) | B (indicbert-v2) | A (mbert-base) | B (mbert-base) | verdict |
|---|---|---|---|---|---|---|
| L1 | linguistic | 4.4% [2.5%–6.7%] | 1.3% [0.2%–2.9%] | 6.4% [4.0%–9.2%] | 2.8% [1.4%–4.5%] | within noise |
| L1 | data | 70.5% [66.5%–74.5%] | 76.0% [72.3%–79.6%] | 68.9% [65.0%–72.6%] | 69.5% [65.6%–73.2%] | within noise |
| L1 | model | 14.2% [11.2%–17.5%] | 12.3% [9.5%–15.3%] | 12.5% [10.0%–15.0%] | 12.2% [9.6%–15.1%] | within noise |
| L1 | unattributed | 10.9% [8.2%–14.0%] | 10.4% [7.8%–13.1%] | 12.2% [9.5%–15.0%] | 15.5% [12.4%–18.5%] | within noise |
| L2 | translation_drift | 3.8% [2.1%–6.0%] | 0.9% [0.0%–2.4%] | **5.2% [3.0%–7.9%]** | **0.9% [0.0%–2.2%]** | one encoder only |
| L2 | tokenizer_fragmentation | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | within noise |
| L2 | morphological_masking | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | within noise |
| L2 | terminology_gap | 0.5% [0.0%–1.5%] | 0.4% [0.0%–0.9%] | 1.2% [0.5%–2.2%] | 1.9% [0.8%–3.2%] | within noise |
| L2 | provenance_specific | 10.6% [8.0%–13.1%] | 9.9% [7.3%–12.4%] | 16.8% [13.9%–20.0%] | 14.1% [11.6%–16.8%] | within noise |
| L2 | cross_lingual_hard | 59.9% [55.6%–64.2%] | 66.1% [62.2%–70.1%] | 52.1% [48.3%–56.1%] | 55.4% [51.3%–59.5%] | within noise |
| L2 | class_collapse | 8.6% [6.0%–11.1%] | 7.9% [5.6%–10.4%] | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | within noise |
| L2 | seed_unstable | 5.6% [3.7%–8.0%] | 4.4% [2.5%–6.5%] | 12.5% [10.0%–15.0%] | 12.2% [9.6%–15.1%] | within noise |

Failures counted — indicbert-v2: A 549, B 546; mbert-base: A 768, B 737. Bold: the two intervals do not overlap in that replicate.

### Task 3: tel target, native vs MT text

A = native tel; B = MT tel. Replicated over encoder: indicbert-v2, mbert-base.

| level | measure | A (indicbert-v2) | B (indicbert-v2) | A (mbert-base) | B (mbert-base) | verdict |
|---|---|---|---|---|---|---|
| L1 | linguistic | 12.2% [9.1%–15.5%] | 9.5% [7.0%–12.5%] | 14.3% [10.8%–18.1%] | 8.5% [5.9%–11.5%] | within noise |
| L1 | data | 73.0% [69.4%–76.8%] | 76.4% [73.0%–79.8%] | 64.2% [60.3%–68.1%] | 66.6% [62.4%–70.5%] | within noise |
| L1 | model | 4.7% [3.2%–6.3%] | 4.1% [2.8%–5.6%] | 10.0% [7.9%–12.4%] | 12.9% [10.4%–15.9%] | within noise |
| L1 | unattributed | 10.2% [7.8%–12.6%] | 9.9% [7.7%–12.1%] | 11.4% [9.1%–13.6%] | 12.0% [9.5%–14.6%] | within noise |
| L2 | translation_drift | **5.8% [3.7%–8.1%]** | **0.6% [0.1%–1.4%]** | **8.1% [5.5%–11.1%]** | **0.8% [0.0%–1.7%]** | held |
| L2 | tokenizer_fragmentation | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | within noise |
| L2 | morphological_masking | 5.2% [3.1%–7.6%] | 7.6% [5.1%–10.6%] | 5.1% [3.0%–7.6%] | 6.9% [4.3%–9.7%] | within noise |
| L2 | terminology_gap | 1.1% [0.5%–1.9%] | 1.3% [0.6%–2.1%] | 1.1% [0.4%–2.1%] | 0.9% [0.3%–1.7%] | within noise |
| L2 | provenance_specific | 7.2% [5.3%–9.3%] | 8.9% [7.0%–10.9%] | 18.9% [15.9%–22.0%] | 17.2% [14.3%–20.3%] | within noise |
| L2 | cross_lingual_hard | 65.7% [61.9%–69.9%] | 67.5% [64.1%–71.1%] | 45.3% [41.4%–49.1%] | 49.4% [45.3%–53.2%] | within noise |
| L2 | class_collapse | **1.8% [0.9%–2.6%]** | **0.0% [0.0%–0.0%]** | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | one encoder only |
| L2 | seed_unstable | 2.9% [1.7%–4.2%] | 4.1% [2.8%–5.6%] | 10.0% [7.9%–12.4%] | 12.9% [10.4%–15.9%] | within noise |

Failures counted — indicbert-v2: A 788, B 798; mbert-base: A 816, B 784. Bold: the two intervals do not overlap in that replicate.

## Without the runs that never trained (CLAUDE7.md hard rule 6)

3 comparison(s) include task-3 IndicBERT-v2 seeds that never trained and were re-run without them.
 These verdicts change:

| comparison | level | measure | all seeds | converged only |
|---|---|---|---|---|
| Task 3: hin target, native vs MT text | L2 | translation_drift | one encoder only | held |
| Task 3: tel target, native vs MT text | L2 | class_collapse | one encoder only | within noise |

