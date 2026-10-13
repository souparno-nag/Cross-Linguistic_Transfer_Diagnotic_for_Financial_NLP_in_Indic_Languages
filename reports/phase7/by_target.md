# Phase 7 — cause mix by target language and typology (T-702)

Config hash `bc5b24734953a7a2`. Shares are of failed (item, seed) rows (CLAUDE7.md). A difference is a finding only if the two 95% CIs do not overlap and the direction agrees in every replicate (hard rule 5).

Two labels carry a stated limitation (hard rule 4):

- `morphological_masking`: a masked term does not raise the failure rate (masked fails more often than intact in 20 of 44 cells; reports/morphology_null.md).
- `terminology_gap`: salience divergence does not separate failures from correct pairs (AUC 0.48-0.52; reports/saliency_baseline.md).

**Scope.** Native-text conditions only. Malayalam has no native split, so its conditions evaluate on MT from another block (`eval_provenance`). Hindi targets cannot receive `morphological_masking` (Hindi writes postpositions as separate words), so a 0% Hindi share is the module's blind spot, not a property of Hindi. Task 3 is at its data ceiling (Phase 4).

## Cause mix by target language

| task | encoder | target | failures | fail rate | unattributed | linguistic | data | model | translation_drift | tokenizer_fragmentation | morphological_masking | terminology_gap | provenance_specific | cross_lingual_hard | class_collapse | seed_unstable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | indicbert-v2 | ben | 1,987 | 30.2% | 15.5% [13.5%–17.4%] | 15.4% [13.1%–17.7%] | 49.1% [46.3%–51.9%] | 20.0% [18.1%–22.0%] | 0.0% | 0.0% | 12.2% | 3.2% | 20.6% | 28.5% | 9.8% | 10.2% |
| 2 | indicbert-v2 | hin | 1,580 | 23.6% | 14.4% [12.4%–16.4%] | 4.3% [3.1%–5.7%] | 41.2% [38.0%–44.5%] | 40.1% [37.3%–43.1%] | 0.0% | 0.0% | 0.0% | 4.3% | 20.8% | 20.4% | 31.1% | 9.0% |
| 2 | indicbert-v2 | mal | 3,011 | 30.3% | 7.8% [6.7%–8.8%] | 10.6% [8.9%–12.1%] | 43.3% [41.0%–45.6%] | 38.4% [36.2%–40.7%] | 7.8% | 0.2% | 0.8% | 1.8% | 15.3% | 28.0% | 32.6% | 5.7% |
| 2 | indicbert-v2 | tel | 2,125 | 34.5% | 0.7% [0.3%–1.2%] | 20.6% [18.0%–23.3%] | 44.3% [41.5%–47.0%] | 34.4% [31.8%–36.8%] | 0.0% | 0.1% | 19.6% | 0.9% | 8.9% | 35.3% | 33.1% | 1.3% |
| 2 | mbert-base | ben | 946 | 14.4% | 13.8% [11.0%–16.9%] | 12.8% [9.8%–16.0%] | 57.0% [53.0%–60.8%] | 16.4% [13.6%–19.4%] | 0.0% | 0.0% | 12.5% | 0.3% | 31.3% | 25.7% | 0.0% | 16.4% |
| 2 | mbert-base | hin | 763 | 11.4% | 13.6% [10.5%–16.8%] | 1.4% [0.6%–2.4%] | 59.9% [55.9%–63.4%] | 25.0% [21.5%–29.0%] | 0.0% | 0.0% | 0.0% | 1.4% | 24.5% | 35.4% | 0.0% | 25.0% |
| 2 | mbert-base | mal | 1,549 | 15.6% | 20.1% [17.2%–22.9%] | 13.4% [11.1%–16.0%] | 45.5% [42.6%–48.4%] | 21.0% [18.7%–23.6%] | 9.9% | 1.4% | 0.6% | 1.4% | 24.1% | 21.4% | 0.0% | 21.0% |
| 2 | mbert-base | tel | 1,147 | 18.6% | 24.0% [20.7%–27.4%] | 18.4% [15.1%–21.6%] | 43.2% [39.6%–47.0%] | 14.4% [12.2%–16.8%] | 0.0% | 0.4% | 17.3% | 0.6% | 23.2% | 20.1% | 0.0% | 14.4% |
| 3 | indicbert-v2 | ben | 701 | 42.6% | 14.6% [11.7%–17.2%] | 13.0% [9.6%–16.5%] | 65.0% [61.1%–69.2%] | 7.4% [5.2%–9.7%] | 6.3% | 0.0% | 3.4% | 3.3% | 8.7% | 56.3% | 0.0% | 7.4% |
| 3 | indicbert-v2 | hin | 549 | 33.4% | 10.9% [8.2%–14.0%] | 4.4% [2.5%–6.7%] | 70.5% [66.5%–74.5%] | 14.2% [11.2%–17.5%] | 3.8% | 0.0% | 0.0% | 0.5% | 10.6% | 59.9% | 8.6% | 5.6% |
| 3 | indicbert-v2 | mal | 992 | 40.2% | 11.0% [9.1%–13.1%] | 14.6% [11.8%–17.6%] | 68.8% [65.8%–71.9%] | 5.6% [4.2%–7.1%] | 13.2% | 0.3% | 0.0% | 1.1% | 9.0% | 59.8% | 0.9% | 4.7% |
| 3 | indicbert-v2 | tel | 788 | 47.9% | 10.2% [7.8%–12.6%] | 12.2% [9.1%–15.5%] | 73.0% [69.4%–76.8%] | 4.7% [3.2%–6.3%] | 5.8% | 0.0% | 5.2% | 1.1% | 7.2% | 65.7% | 1.8% | 2.9% |
| 3 | mbert-base | ben | 802 | 48.8% | 11.1% [8.4%–13.9%] | 13.6% [10.3%–17.1%] | 64.8% [61.0%–68.9%] | 10.5% [8.1%–12.9%] | 8.7% | 0.2% | 2.9% | 1.7% | 19.1% | 45.8% | 0.0% | 10.5% |
| 3 | mbert-base | hin | 768 | 46.7% | 12.2% [9.5%–15.0%] | 6.4% [4.0%–9.2%] | 68.9% [65.0%–72.6%] | 12.5% [10.0%–15.0%] | 5.2% | 0.0% | 0.0% | 1.2% | 16.8% | 52.1% | 0.0% | 12.5% |
| 3 | mbert-base | mal | 1,254 | 50.9% | 15.8% [13.3%–18.4%] | 16.6% [13.6%–19.7%] | 58.1% [54.7%–61.4%] | 9.6% [7.9%–11.3%] | 14.5% | 1.4% | 0.0% | 0.7% | 13.5% | 44.6% | 0.0% | 9.6% |
| 3 | mbert-base | tel | 816 | 49.6% | 11.4% [9.1%–13.6%] | 14.3% [10.8%–18.1%] | 64.2% [60.3%–68.1%] | 10.0% [7.9%–12.4%] | 8.1% | 0.0% | 5.1% | 1.1% | 18.9% | 45.3% | 0.0% | 10.0% |

## Cause mix by typological quadrant

| task | encoder | quadrant | failures | fail rate | unattributed | linguistic | data | model | translation_drift | tokenizer_fragmentation | morphological_masking | terminology_gap | provenance_specific | cross_lingual_hard | class_collapse | seed_unstable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | indicbert-v2 | Dravidian->Dravidian | 1,061 | 32.2% | 0.3% [0.0%–0.7%] | 10.0% [7.1%–12.8%] | 20.5% [17.7%–23.4%] | 69.3% [65.9%–72.9%] | 7.4% | 0.0% | 0.9% | 1.6% | 13.4% | 7.1% | 68.4% | 0.8% |
| 2 | indicbert-v2 | Dravidian->Indo-Aryan | 1,742 | 26.2% | 17.2% [14.9%–19.2%] | 11.4% [9.5%–13.5%] | 31.2% [28.6%–33.8%] | 40.2% [37.5%–43.1%] | 0.0% | 0.0% | 6.4% | 5.0% | 23.8% | 7.5% | 28.2% | 12.0% |
| 2 | indicbert-v2 | Indo-Aryan->Dravidian | 4,075 | 31.8% | 6.0% [5.2%–6.9%] | 16.0% [14.3%–17.6%] | 49.7% [47.6%–51.7%] | 28.3% [26.5%–29.9%] | 3.8% | 0.2% | 10.6% | 1.3% | 12.5% | 37.3% | 23.6% | 4.7% |
| 2 | indicbert-v2 | Indo-Aryan->Indo-Aryan | 1,825 | 27.5% | 12.9% [11.0%–14.7%] | 9.6% [7.7%–11.6%] | 59.3% [56.4%–62.1%] | 18.1% [16.1%–20.0%] | 0.0% | 0.0% | 7.2% | 2.4% | 17.7% | 41.6% | 10.7% | 7.5% |
| 2 | mbert-base | Dravidian->Dravidian | 466 | 14.1% | 10.5% [6.9%–14.3%] | 12.9% [8.8%–17.4%] | 53.2% [47.6%–58.7%] | 23.4% [19.3%–28.1%] | 10.3% | 0.6% | 0.9% | 1.1% | 28.5% | 24.7% | 0.0% | 23.4% |
| 2 | mbert-base | Dravidian->Indo-Aryan | 986 | 14.8% | 15.5% [12.4%–18.4%] | 8.0% [5.6%–10.7%] | 51.5% [47.8%–55.3%] | 24.9% [21.6%–28.4%] | 0.0% | 0.0% | 7.6% | 0.4% | 22.1% | 29.4% | 0.0% | 24.9% |
| 2 | mbert-base | Indo-Aryan->Dravidian | 2,230 | 17.4% | 24.1% [21.6%–26.5%] | 16.1% [14.0%–18.3%] | 42.7% [40.2%–45.5%] | 17.1% [15.3%–19.1%] | 4.8% | 1.1% | 9.1% | 1.1% | 22.7% | 20.0% | 0.0% | 17.1% |
| 2 | mbert-base | Indo-Aryan->Indo-Aryan | 723 | 10.9% | 11.3% [8.2%–14.6%] | 7.3% [4.9%–10.1%] | 67.5% [63.6%–71.4%] | 13.8% [11.1%–16.6%] | 0.0% | 0.0% | 5.9% | 1.4% | 36.7% | 30.8% | 0.0% | 13.8% |
| 3 | indicbert-v2 | Dravidian->Dravidian | 204 | 24.8% | 14.2% [9.5%–19.5%] | 15.7% [10.1%–21.7%] | 58.3% [51.4%–65.3%] | 11.8% [7.4%–16.3%] | 15.7% | 0.0% | 0.0% | 0.0% | 11.3% | 47.1% | 4.4% | 7.4% |
| 3 | indicbert-v2 | Dravidian->Indo-Aryan | 440 | 26.8% | 18.0% [14.3%–22.2%] | 9.8% [6.0%–13.4%] | 57.7% [52.9%–62.8%] | 14.5% [10.8%–18.6%] | 8.2% | 0.0% | 1.6% | 0.0% | 12.7% | 45.0% | 0.0% | 14.5% |
| 3 | indicbert-v2 | Indo-Aryan->Dravidian | 1,576 | 47.9% | 10.2% [8.7%–11.8%] | 13.3% [10.8%–15.4%] | 72.2% [69.6%–74.7%] | 4.4% [3.3%–5.4%] | 9.2% | 0.2% | 2.6% | 1.3% | 7.8% | 64.4% | 0.9% | 3.5% |
| 3 | indicbert-v2 | Indo-Aryan->Indo-Aryan | 810 | 49.3% | 10.2% [8.1%–12.6%] | 8.9% [6.4%–11.5%] | 72.7% [69.1%–76.0%] | 8.1% [6.1%–10.1%] | 3.6% | 0.0% | 2.1% | 3.2% | 7.8% | 64.9% | 5.8% | 2.3% |
| 3 | mbert-base | Dravidian->Dravidian | 411 | 50.0% | 14.8% [10.1%–19.9%] | 17.0% [11.9%–22.8%] | 60.1% [53.5%–66.2%] | 8.0% [5.4%–10.7%] | 17.0% | 0.0% | 0.0% | 0.0% | 16.3% | 43.8% | 0.0% | 8.0% |
| 3 | mbert-base | Dravidian->Indo-Aryan | 811 | 49.3% | 12.9% [10.2%–15.7%] | 10.1% [7.1%–13.4%] | 65.5% [62.0%–69.1%] | 11.5% [9.2%–13.8%] | 8.1% | 0.0% | 1.2% | 0.7% | 18.4% | 47.1% | 0.0% | 11.5% |
| 3 | mbert-base | Indo-Aryan->Dravidian | 1,659 | 50.5% | 13.9% [11.9%–15.8%] | 15.4% [12.7%–18.1%] | 60.6% [57.7%–63.5%] | 10.2% [8.6%–11.8%] | 10.7% | 1.0% | 2.5% | 1.1% | 15.4% | 45.1% | 0.0% | 10.2% |
| 3 | mbert-base | Indo-Aryan->Indo-Aryan | 759 | 46.2% | 10.3% [7.8%–12.8%] | 10.0% [7.3%–13.3%] | 68.2% [64.2%–71.9%] | 11.5% [9.3%–14.0%] | 5.8% | 0.3% | 1.7% | 2.2% | 17.5% | 50.7% | 0.0% | 11.5% |

## Transfer gap beside the cause mix

Mean macro-F1 gap (source − target) per target, from `reports/source_comparison.parquet` (T-408, which leaves out seeds that never trained). Context only; not recomputed here.

| task | encoder | target | mean gap | sources |
|---|---|---|---|---|
| 2 | indicbert-v2 | ben | 0.386 | hin, tel |
| 2 | indicbert-v2 | hin | 0.312 | ben, tel |
| 2 | indicbert-v2 | mal | 0.390 | ben, hin, tel |
| 2 | indicbert-v2 | tel | 0.428 | ben, hin |
| 2 | mbert-base | ben | 0.169 | hin, tel |
| 2 | mbert-base | hin | 0.132 | ben, tel |
| 2 | mbert-base | mal | 0.185 | ben, hin, tel |
| 2 | mbert-base | tel | 0.206 | ben, hin |
| 3 | indicbert-v2 | ben | 0.494 | hin, tel |
| 3 | indicbert-v2 | hin | 0.500 | ben, tel |
| 3 | indicbert-v2 | mal | 0.491 | ben, hin, tel |
| 3 | indicbert-v2 | tel | 0.509 | ben, hin |
| 3 | mbert-base | ben | 0.520 | hin, tel |
| 3 | mbert-base | hin | 0.491 | ben, tel |
| 3 | mbert-base | mal | 0.542 | ben, hin, tel |
| 3 | mbert-base | tel | 0.519 | ben, hin |

## Comparisons

### Task 2: into Dravidian vs out of it (hin↔tel, ben↔tel)

A = hin→tel + ben→tel; B = tel→hin + tel→ben. Replicated over encoder: indicbert-v2, mbert-base.

| level | measure | A (indicbert-v2) | B (indicbert-v2) | A (mbert-base) | B (mbert-base) | verdict |
|---|---|---|---|---|---|---|
| L1 | linguistic | **20.6% [18.0%–23.3%]** | **11.4% [9.5%–13.5%]** | **18.4% [15.1%–21.6%]** | **8.0% [5.6%–10.7%]** | held |
| L1 | data | **44.3% [41.5%–47.0%]** | **31.2% [28.6%–33.8%]** | **43.2% [39.6%–47.0%]** | **51.5% [47.8%–55.3%]** | encoders disagree |
| L1 | model | **34.4% [31.8%–36.8%]** | **40.2% [37.5%–43.1%]** | **14.4% [12.2%–16.8%]** | **24.9% [21.6%–28.4%]** | held |
| L1 | unattributed | **0.7% [0.3%–1.2%]** | **17.2% [14.9%–19.2%]** | **24.0% [20.7%–27.4%]** | **15.5% [12.4%–18.4%]** | encoders disagree |
| L2 | translation_drift | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | within noise |
| L2 | tokenizer_fragmentation | 0.1% [0.0%–0.3%] | 0.0% [0.0%–0.0%] | 0.4% [0.0%–1.2%] | 0.0% [0.0%–0.0%] | within noise |
| L2 | morphological_masking | **19.6% [17.1%–22.3%]** | **6.4% [4.9%–8.2%]** | **17.3% [14.1%–20.5%]** | **7.6% [5.2%–10.2%]** | held |
| L2 | terminology_gap | **0.9% [0.4%–1.5%]** | **5.0% [3.7%–6.4%]** | 0.6% [0.1%–1.2%] | 0.4% [0.1%–0.8%] | one encoder only |
| L2 | provenance_specific | **8.9% [7.6%–10.5%]** | **23.8% [21.4%–26.2%]** | 23.2% [20.0%–26.5%] | 22.1% [19.3%–25.2%] | one encoder only |
| L2 | cross_lingual_hard | **35.3% [32.6%–38.1%]** | **7.5% [6.0%–9.0%]** | **20.1% [17.2%–23.2%]** | **29.4% [26.0%–32.9%]** | encoders disagree |
| L2 | class_collapse | 33.1% [30.4%–35.6%] | 28.2% [25.6%–30.9%] | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | within noise |
| L2 | seed_unstable | **1.3% [0.8%–1.9%]** | **12.0% [10.3%–13.9%]** | **14.4% [12.2%–16.8%]** | **24.9% [21.6%–28.4%]** | held |

Failures counted — indicbert-v2: A 2,125, B 1,742; mbert-base: A 1,147, B 986. Bold: the two intervals do not overlap in that replicate.

### Task 2: hin→tel vs tel→hin

A = hin→tel; B = tel→hin. Replicated over encoder: indicbert-v2, mbert-base.

| level | measure | A (indicbert-v2) | B (indicbert-v2) | A (mbert-base) | B (mbert-base) | verdict |
|---|---|---|---|---|---|---|
| L1 | linguistic | **20.2% [15.9%–24.0%]** | **4.8% [3.2%–6.7%]** | **18.9% [14.7%–23.5%]** | **0.2% [0.0%–0.7%]** | held |
| L1 | data | **49.3% [45.3%–53.1%]** | **23.2% [19.9%–26.3%]** | **39.1% [34.5%–43.8%]** | **52.9% [47.8%–58.2%]** | encoders disagree |
| L1 | model | **30.0% [26.6%–33.4%]** | **61.4% [58.2%–64.5%]** | **15.1% [11.7%–18.6%]** | **30.0% [24.7%–35.4%]** | held |
| L1 | unattributed | **0.5% [0.0%–1.0%]** | **10.6% [8.9%–12.4%]** | **26.9% [22.1%–31.6%]** | **16.9% [12.5%–21.4%]** | encoders disagree |
| L2 | translation_drift | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | within noise |
| L2 | tokenizer_fragmentation | 0.2% [0.0%–0.5%] | 0.0% [0.0%–0.0%] | 0.8% [0.0%–2.1%] | 0.0% [0.0%–0.0%] | within noise |
| L2 | morphological_masking | **19.1% [15.0%–22.9%]** | **0.0% [0.0%–0.0%]** | **17.5% [13.4%–21.9%]** | **0.0% [0.0%–0.0%]** | held |
| L2 | terminology_gap | **0.9% [0.4%–1.5%]** | **4.8% [3.2%–6.7%]** | 0.6% [0.0%–1.5%] | 0.2% [0.0%–0.7%] | one encoder only |
| L2 | provenance_specific | **6.2% [4.6%–7.9%]** | **15.5% [12.9%–18.2%]** | 20.4% [16.7%–24.4%] | 17.8% [14.1%–22.0%] | one encoder only |
| L2 | cross_lingual_hard | **43.2% [39.5%–47.0%]** | **7.7% [5.7%–9.9%]** | **18.7% [14.8%–22.9%]** | **35.1% [29.9%–40.6%]** | encoders disagree |
| L2 | class_collapse | **28.7% [25.3%–32.2%]** | **57.8% [54.8%–61.0%]** | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | one encoder only |
| L2 | seed_unstable | **1.3% [0.6%–2.0%]** | **3.5% [2.3%–5.0%]** | **15.1% [11.7%–18.6%]** | **30.0% [24.7%–35.4%]** | held |

Failures counted — indicbert-v2: A 1,103, B 849; mbert-base: A 647, B 450. Bold: the two intervals do not overlap in that replicate.

### Task 2: ben→tel vs tel→ben

A = ben→tel; B = tel→ben. Replicated over encoder: indicbert-v2, mbert-base.

| level | measure | A (indicbert-v2) | B (indicbert-v2) | A (mbert-base) | B (mbert-base) | verdict |
|---|---|---|---|---|---|---|
| L1 | linguistic | 20.9% [17.0%–24.9%] | 17.7% [14.0%–21.0%] | 17.8% [13.3%–23.4%] | 14.6% [10.4%–19.0%] | within noise |
| L1 | data | 38.8% [35.0%–42.9%] | 38.9% [35.3%–42.4%] | 48.6% [42.4%–53.8%] | 50.4% [44.8%–55.7%] | within noise |
| L1 | model | **39.2% [35.4%–43.1%]** | **20.0% [17.0%–23.3%]** | 13.4% [10.1%–17.0%] | 20.7% [16.6%–25.1%] | one encoder only |
| L1 | unattributed | **1.0% [0.3%–1.9%]** | **23.4% [19.7%–27.1%]** | 20.2% [15.4%–25.3%] | 14.4% [10.5%–18.4%] | one encoder only |
| L2 | translation_drift | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | within noise |
| L2 | tokenizer_fragmentation | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | within noise |
| L2 | morphological_masking | **20.1% [16.0%–24.1%]** | **12.5% [9.5%–15.8%]** | 17.2% [12.7%–22.5%] | 14.0% [9.9%–18.5%] | one encoder only |
| L2 | terminology_gap | **0.9% [0.1%–1.9%]** | **5.2% [3.4%–7.0%]** | 0.6% [0.0%–1.5%] | 0.6% [0.0%–1.2%] | one encoder only |
| L2 | provenance_specific | **11.9% [9.7%–14.2%]** | **31.6% [27.9%–35.0%]** | 26.8% [21.4%–32.2%] | 25.7% [21.3%–30.3%] | one encoder only |
| L2 | cross_lingual_hard | **26.9% [23.3%–30.6%]** | **7.3% [5.5%–9.3%]** | 21.8% [16.7%–26.3%] | 24.6% [20.5%–29.1%] | one encoder only |
| L2 | class_collapse | **37.9% [33.8%–41.7%]** | **0.0% [0.0%–0.0%]** | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | one encoder only |
| L2 | seed_unstable | **1.4% [0.7%–2.3%]** | **20.0% [17.0%–23.3%]** | 13.4% [10.1%–17.0%] | 20.7% [16.6%–25.1%] | one encoder only |

Failures counted — indicbert-v2: A 1,022, B 893; mbert-base: A 500, B 536. Bold: the two intervals do not overlap in that replicate.

### Task 3: into Dravidian vs out of it (hin↔tel, ben↔tel)

A = hin→tel + ben→tel; B = tel→hin + tel→ben. Replicated over encoder: indicbert-v2, mbert-base.

| level | measure | A (indicbert-v2) | B (indicbert-v2) | A (mbert-base) | B (mbert-base) | verdict |
|---|---|---|---|---|---|---|
| L1 | linguistic | 12.2% [9.1%–15.5%] | 9.8% [6.0%–13.4%] | 14.3% [10.8%–18.1%] | 10.1% [7.1%–13.4%] | within noise |
| L1 | data | **73.0% [69.4%–76.8%]** | **57.7% [52.9%–62.8%]** | 64.2% [60.3%–68.1%] | 65.5% [62.0%–69.1%] | one encoder only |
| L1 | model | **4.7% [3.2%–6.3%]** | **14.5% [10.8%–18.6%]** | 10.0% [7.9%–12.4%] | 11.5% [9.2%–13.8%] | one encoder only |
| L1 | unattributed | **10.2% [7.8%–12.6%]** | **18.0% [14.3%–22.2%]** | 11.4% [9.1%–13.6%] | 12.9% [10.2%–15.7%] | one encoder only |
| L2 | translation_drift | 5.8% [3.7%–8.1%] | 8.2% [4.9%–11.5%] | 8.1% [5.5%–11.1%] | 8.1% [5.4%–11.0%] | within noise |
| L2 | tokenizer_fragmentation | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | within noise |
| L2 | morphological_masking | **5.2% [3.1%–7.6%]** | **1.6% [0.4%–3.1%]** | **5.1% [3.0%–7.6%]** | **1.2% [0.3%–2.5%]** | held |
| L2 | terminology_gap | **1.1% [0.5%–1.9%]** | **0.0% [0.0%–0.0%]** | 1.1% [0.4%–2.1%] | 0.7% [0.1%–1.5%] | one encoder only |
| L2 | provenance_specific | **7.2% [5.3%–9.3%]** | **12.7% [9.5%–15.9%]** | 18.9% [15.9%–22.0%] | 18.4% [15.4%–21.4%] | one encoder only |
| L2 | cross_lingual_hard | **65.7% [61.9%–69.9%]** | **45.0% [40.4%–50.1%]** | 45.3% [41.4%–49.1%] | 47.1% [43.0%–50.9%] | one encoder only |
| L2 | class_collapse | **1.8% [0.9%–2.6%]** | **0.0% [0.0%–0.0%]** | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | one encoder only |
| L2 | seed_unstable | **2.9% [1.7%–4.2%]** | **14.5% [10.8%–18.6%]** | 10.0% [7.9%–12.4%] | 11.5% [9.2%–13.8%] | one encoder only |

Failures counted — indicbert-v2: A 788, B 440; mbert-base: A 816, B 811. Bold: the two intervals do not overlap in that replicate.

### Task 3: hin→tel vs tel→hin

A = hin→tel; B = tel→hin. Replicated over encoder: indicbert-v2, mbert-base.

| level | measure | A (indicbert-v2) | B (indicbert-v2) | A (mbert-base) | B (mbert-base) | verdict |
|---|---|---|---|---|---|---|
| L1 | linguistic | 10.5% [7.0%–14.3%] | 4.2% [1.0%–8.1%] | 12.3% [8.4%–16.8%] | 5.4% [2.6%–8.7%] | within noise |
| L1 | data | 74.7% [69.9%–79.3%] | 64.2% [58.0%–70.9%] | 67.2% [62.2%–72.3%] | 69.6% [64.6%–74.4%] | within noise |
| L1 | model | **3.2% [1.7%–5.2%]** | **13.7% [8.4%–19.1%]** | 9.9% [6.9%–13.2%] | 11.0% [7.9%–14.6%] | one encoder only |
| L1 | unattributed | 11.6% [8.6%–14.8%] | 17.9% [12.9%–23.1%] | 10.6% [7.5%–13.7%] | 14.0% [9.7%–18.0%] | within noise |
| L2 | translation_drift | 3.0% [1.1%–5.2%] | 4.2% [1.0%–8.1%] | 4.4% [2.0%–7.5%] | 4.3% [1.7%–7.5%] | within noise |
| L2 | tokenizer_fragmentation | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | within noise |
| L2 | morphological_masking | **5.6% [2.7%–9.0%]** | **0.0% [0.0%–0.0%]** | **5.9% [2.8%–9.7%]** | **0.0% [0.0%–0.0%]** | held |
| L2 | terminology_gap | **1.9% [0.9%–3.2%]** | **0.0% [0.0%–0.0%]** | 2.0% [0.5%–3.8%] | 1.0% [0.0%–2.5%] | one encoder only |
| L2 | provenance_specific | **7.9% [5.5%–10.5%]** | **16.5% [11.5%–21.6%]** | 19.7% [15.8%–23.7%] | 18.6% [14.7%–22.8%] | one encoder only |
| L2 | cross_lingual_hard | **66.7% [61.7%–71.6%]** | **47.6% [41.0%–54.5%]** | 47.5% [42.4%–52.9%] | 51.0% [44.9%–56.8%] | one encoder only |
| L2 | class_collapse | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | within noise |
| L2 | seed_unstable | **3.2% [1.7%–5.2%]** | **13.7% [8.4%–19.1%]** | 9.9% [6.9%–13.2%] | 11.0% [7.9%–14.6%] | one encoder only |

Failures counted — indicbert-v2: A 466, B 212; mbert-base: A 406, B 392. Bold: the two intervals do not overlap in that replicate.

### Task 3: ben→tel vs tel→ben

A = ben→tel; B = tel→ben. Replicated over encoder: indicbert-v2, mbert-base.

| level | measure | A (indicbert-v2) | B (indicbert-v2) | A (mbert-base) | B (mbert-base) | verdict |
|---|---|---|---|---|---|---|
| L1 | linguistic | 14.6% [9.9%–19.7%] | 14.9% [9.1%–21.8%] | 16.3% [11.2%–22.1%] | 14.6% [9.4%–19.9%] | within noise |
| L1 | data | **70.5% [64.7%–76.2%]** | **51.8% [44.6%–58.9%]** | 61.2% [55.7%–67.0%] | 61.6% [56.1%–67.5%] | one encoder only |
| L1 | model | 6.8% [4.3%–9.8%] | 15.4% [9.7%–20.9%] | 10.2% [7.3%–13.4%] | 11.9% [8.8%–15.5%] | within noise |
| L1 | unattributed | **8.1% [4.8%–11.3%]** | **18.0% [12.8%–23.3%]** | 12.2% [8.7%–16.1%] | 11.9% [8.3%–15.7%] | one encoder only |
| L2 | translation_drift | 9.9% [5.8%–14.4%] | 11.8% [6.5%–18.0%] | 11.7% [7.2%–16.6%] | 11.7% [7.1%–16.4%] | within noise |
| L2 | tokenizer_fragmentation | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | within noise |
| L2 | morphological_masking | 4.7% [1.9%–7.7%] | 3.1% [0.5%–6.4%] | 4.4% [1.7%–7.9%] | 2.4% [0.5%–4.9%] | within noise |
| L2 | terminology_gap | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | 0.2% [0.0%–0.8%] | 0.5% [0.0%–1.2%] | within noise |
| L2 | provenance_specific | 6.2% [3.6%–9.1%] | 9.2% [5.1%–13.6%] | 18.0% [13.8%–22.4%] | 18.1% [14.0%–22.7%] | within noise |
| L2 | cross_lingual_hard | **64.3% [58.8%–69.9%]** | **42.5% [35.7%–49.8%]** | 43.2% [37.6%–48.8%] | 43.4% [37.9%–49.3%] | one encoder only |
| L2 | class_collapse | **4.3% [2.3%–6.6%]** | **0.0% [0.0%–0.0%]** | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | one encoder only |
| L2 | seed_unstable | **2.5% [0.9%–4.3%]** | **15.4% [9.7%–20.9%]** | 10.2% [7.3%–13.4%] | 11.9% [8.8%–15.5%] | one encoder only |

Failures counted — indicbert-v2: A 322, B 228; mbert-base: A 410, B 419. Bold: the two intervals do not overlap in that replicate.

## Without the runs that never trained (CLAUDE7.md hard rule 6)

3 comparison(s) include task-3 IndicBERT-v2 seeds that never trained and were re-run without them.
 These verdicts change:

| comparison | level | measure | all seeds | converged only |
|---|---|---|---|---|
| Task 3: into Dravidian vs out of it (hin↔tel, ben↔tel) | L1 | unattributed | one encoder only | within noise |
| Task 3: into Dravidian vs out of it (hin↔tel, ben↔tel) | L2 | provenance_specific | one encoder only | within noise |
| Task 3: into Dravidian vs out of it (hin↔tel, ben↔tel) | L2 | class_collapse | one encoder only | within noise |
| Task 3: hin→tel vs tel→hin | L1 | linguistic | within noise | one encoder only |
| Task 3: hin→tel vs tel→hin | L1 | model | one encoder only | within noise |
| Task 3: hin→tel vs tel→hin | L2 | provenance_specific | one encoder only | within noise |
| Task 3: hin→tel vs tel→hin | L2 | cross_lingual_hard | one encoder only | within noise |
| Task 3: hin→tel vs tel→hin | L2 | seed_unstable | one encoder only | within noise |
| Task 3: ben→tel vs tel→ben | L1 | model | within noise | one encoder only |
| Task 3: ben→tel vs tel→ben | L1 | unattributed | one encoder only | within noise |
| Task 3: ben→tel vs tel→ben | L2 | class_collapse | one encoder only | within noise |

