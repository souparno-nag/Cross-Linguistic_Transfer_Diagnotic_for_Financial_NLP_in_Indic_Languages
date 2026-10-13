# Phase 7 — cause mix by source language (T-703)

Config hash `bc5b24734953a7a2`. Shares are of failed (item, seed) rows (CLAUDE7.md). A difference is a finding only if the two 95% CIs do not overlap and the direction agrees in every replicate (hard rule 5).

Two labels carry a stated limitation (hard rule 4):

- `morphological_masking`: a masked term does not raise the failure rate (masked fails more often than intact in 20 of 44 cells; reports/morphology_null.md).
- `terminology_gap`: salience divergence does not separate failures from correct pairs (AUC 0.48-0.52; reports/saliency_baseline.md).

**Scope.** Native-text conditions only. The table below pools each source over all three of its targets, so it mixes target effects into source effects; the comparisons further down do not, because each pair of sources is compared on the targets both reach. `fail rate` is the volume half of the question: failed rows over evaluated rows.

## Cause mix by source language (all targets pooled)

| task | encoder | source | failures | fail rate | unattributed | linguistic | data | model | translation_drift | tokenizer_fragmentation | morphological_masking | terminology_gap | provenance_specific | cross_lingual_hard | class_collapse | seed_unstable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | indicbert-v2 | ben | 2,714 | 27.7% | 11.4% [9.8%–12.9%] | 12.5% [10.6%–14.3%] | 47.8% [45.3%–50.1%] | 28.3% [26.3%–30.4%] | 2.7% | 0.3% | 7.7% | 1.8% | 18.2% | 29.6% | 19.6% | 8.7% |
| 2 | indicbert-v2 | hin | 3,186 | 32.9% | 5.4% [4.7%–6.1%] | 15.3% [13.4%–17.1%] | 56.9% [54.5%–59.0%] | 22.4% [20.9%–24.1%] | 2.6% | 0.1% | 11.0% | 1.6% | 10.5% | 46.4% | 19.6% | 2.8% |
| 2 | indicbert-v2 | tel | 2,803 | 28.2% | 10.8% [9.3%–12.3%] | 10.9% [9.2%–12.5%] | 27.1% [25.1%–29.0%] | 51.2% [48.7%–53.5%] | 2.8% | 0.0% | 4.4% | 3.7% | 19.8% | 7.3% | 43.4% | 7.8% |
| 2 | mbert-base | ben | 1,309 | 13.4% | 17.8% [15.2%–20.6%] | 12.7% [10.1%–15.3%] | 51.5% [47.9%–54.9%] | 18.0% [15.4%–20.6%] | 3.9% | 0.5% | 6.6% | 1.7% | 26.4% | 25.1% | 0.0% | 18.0% |
| 2 | mbert-base | hin | 1,644 | 17.0% | 23.5% [20.7%–26.6%] | 14.9% [12.5%–17.6%] | 46.7% [43.4%–49.8%] | 14.9% [12.9%–17.0%] | 3.3% | 1.0% | 9.8% | 0.7% | 25.9% | 20.7% | 0.0% | 14.9% |
| 2 | mbert-base | tel | 1,452 | 14.6% | 13.9% [11.6%–16.3%] | 9.6% [7.3%–12.0%] | 52.1% [49.1%–55.2%] | 24.4% [21.8%–27.0%] | 3.3% | 0.2% | 5.4% | 0.6% | 24.2% | 27.9% | 0.0% | 24.4% |
| 3 | indicbert-v2 | ben | 991 | 40.2% | 8.1% [6.3%–9.9%] | 11.2% [8.7%–13.7%] | 72.1% [69.0%–75.3%] | 8.6% [6.9%–10.3%] | 9.0% | 0.2% | 1.5% | 0.5% | 7.7% | 64.5% | 6.2% | 2.4% |
| 3 | indicbert-v2 | hin | 1,395 | 56.6% | 11.7% [10.0%–13.5%] | 12.2% [9.8%–14.5%] | 72.5% [69.9%–75.2%] | 3.6% [2.6%–4.7%] | 6.1% | 0.1% | 3.1% | 2.9% | 7.9% | 64.7% | 0.0% | 3.6% |
| 3 | indicbert-v2 | tel | 644 | 26.1% | 16.8% [13.7%–20.1%] | 11.6% [8.9%–14.6%] | 57.9% [53.9%–62.0%] | 13.7% [10.7%–16.9%] | 10.6% | 0.0% | 1.1% | 0.0% | 12.3% | 45.7% | 1.4% | 12.3% |
| 3 | mbert-base | ben | 1,205 | 48.9% | 13.5% [11.3%–15.7%] | 13.3% [10.6%–16.5%] | 61.7% [57.9%–65.0%] | 11.5% [9.7%–13.5%] | 11.0% | 0.2% | 1.5% | 0.7% | 15.3% | 46.4% | 0.0% | 11.5% |
| 3 | mbert-base | hin | 1,213 | 49.2% | 12.0% [9.8%–14.1%] | 14.1% [11.3%–16.8%] | 64.3% [61.1%–67.5%] | 9.6% [7.9%–11.6%] | 7.4% | 1.4% | 3.1% | 2.2% | 16.9% | 47.4% | 0.0% | 9.6% |
| 3 | mbert-base | tel | 1,222 | 49.6% | 13.6% [11.2%–16.0%] | 12.4% [9.8%–15.0%] | 63.7% [60.2%–66.7%] | 10.3% [8.5%–12.3%] | 11.1% | 0.0% | 0.8% | 0.5% | 17.7% | 46.0% | 0.0% | 10.3% |

## Comparisons

### Task 2: hin source vs ben source, on targets mal+tel

A = hin→{mal,tel}; B = ben→{mal,tel}. Replicated over encoder: indicbert-v2, mbert-base.

| level | measure | A (indicbert-v2) | B (indicbert-v2) | A (mbert-base) | B (mbert-base) | verdict |
|---|---|---|---|---|---|---|
| L1 | linguistic | 16.2% [14.0%–18.8%] | 15.7% [13.3%–18.2%] | 16.4% [13.2%–19.5%] | 15.7% [12.4%–19.0%] | within noise |
| L1 | data | **56.6% [53.8%–59.5%]** | **42.5% [39.6%–45.2%]** | 40.4% [36.9%–43.8%] | 45.7% [41.5%–49.7%] | one encoder only |
| L1 | model | **23.7% [21.5%–25.9%]** | **33.1% [30.8%–35.8%]** | 16.3% [13.9%–18.9%] | 18.1% [15.1%–21.3%] | one encoder only |
| L1 | unattributed | **3.5% [2.8%–4.4%]** | **8.7% [7.2%–10.2%]** | 27.0% [23.8%–30.5%] | 20.6% [17.2%–24.3%] | one encoder only |
| L2 | translation_drift | 4.0% [2.7%–5.2%] | 3.7% [2.5%–4.9%] | 4.5% [2.8%–6.3%] | 5.1% [3.2%–7.3%] | within noise |
| L2 | tokenizer_fragmentation | 0.1% [0.0%–0.3%] | 0.4% [0.1%–0.7%] | 1.4% [0.5%–2.5%] | 0.7% [0.0%–1.6%] | within noise |
| L2 | morphological_masking | 10.6% [8.6%–12.8%] | 10.5% [8.5%–12.8%] | 9.6% [7.2%–12.0%] | 8.6% [6.1%–11.3%] | within noise |
| L2 | terminology_gap | 1.6% [1.0%–2.2%] | 1.1% [0.5%–1.8%] | 1.0% [0.4%–1.6%] | 1.2% [0.4%–2.2%] | within noise |
| L2 | provenance_specific | **10.0% [8.5%–11.4%]** | **15.1% [13.1%–17.0%]** | 21.7% [18.8%–24.6%] | 24.0% [20.7%–27.5%] | one encoder only |
| L2 | cross_lingual_hard | **46.6% [43.9%–49.4%]** | **27.4% [24.8%–30.0%]** | 18.6% [16.0%–21.5%] | 21.7% [18.3%–25.0%] | one encoder only |
| L2 | class_collapse | **20.5% [18.4%–22.8%]** | **26.8% [24.6%–29.5%]** | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | one encoder only |
| L2 | seed_unstable | **3.2% [2.4%–4.0%]** | **6.3% [5.2%–7.5%]** | 16.3% [13.9%–18.9%] | 18.1% [15.1%–21.3%] | one encoder only |

Failures counted — indicbert-v2: A 2,092, B 1,983; mbert-base: A 1,234, B 996. Bold: the two intervals do not overlap in that replicate.

### Task 2: hin source vs tel source, on targets ben+mal

A = hin→{ben,mal}; B = tel→{ben,mal}. Replicated over encoder: indicbert-v2, mbert-base.

| level | measure | A (indicbert-v2) | B (indicbert-v2) | A (mbert-base) | B (mbert-base) | verdict |
|---|---|---|---|---|---|---|
| L1 | linguistic | 12.7% [10.5%–14.8%] | 13.5% [11.4%–15.6%] | 12.3% [9.3%–15.3%] | 13.8% [10.8%–17.0%] | within noise |
| L1 | data | **60.9% [58.1%–63.5%]** | **28.9% [26.6%–31.3%]** | 51.6% [47.4%–55.7%] | 51.7% [47.9%–55.5%] | one encoder only |
| L1 | model | **18.4% [16.4%–20.4%]** | **46.8% [43.9%–49.9%]** | **14.7% [12.1%–17.7%]** | **22.0% [18.9%–25.1%]** | held |
| L1 | unattributed | 8.0% [6.9%–9.2%] | 10.8% [8.9%–12.7%] | **21.4% [17.8%–25.2%]** | **12.6% [9.9%–15.4%]** | one encoder only |
| L2 | translation_drift | 4.0% [2.9%–5.3%] | 4.0% [2.6%–5.5%] | 5.5% [3.4%–7.9%] | 4.8% [2.9%–6.9%] | within noise |
| L2 | tokenizer_fragmentation | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | 1.2% [0.3%–2.3%] | 0.3% [0.0%–0.8%] | within noise |
| L2 | morphological_masking | 6.8% [5.0%–8.5%] | 6.2% [4.7%–7.9%] | 4.8% [3.0%–6.7%] | 7.9% [5.2%–10.3%] | within noise |
| L2 | terminology_gap | 1.9% [1.3%–2.6%] | 3.2% [2.4%–4.2%] | 0.8% [0.3%–1.5%] | 0.8% [0.3%–1.4%] | within noise |
| L2 | provenance_specific | **12.9% [11.3%–14.5%]** | **21.7% [19.6%–24.0%]** | 29.5% [25.8%–33.1%] | 27.0% [24.0%–30.4%] | one encoder only |
| L2 | cross_lingual_hard | **48.1% [45.4%–50.7%]** | **7.2% [5.9%–8.4%]** | 22.1% [18.7%–25.4%] | 24.7% [21.2%–28.3%] | one encoder only |
| L2 | class_collapse | **14.7% [13.0%–16.7%]** | **37.2% [34.1%–40.3%]** | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | one encoder only |
| L2 | seed_unstable | **3.6% [2.8%–4.5%]** | **9.6% [8.1%–11.3%]** | **14.7% [12.1%–17.7%]** | **22.0% [18.9%–25.1%]** | held |

Failures counted — indicbert-v2: A 2,083, B 1,954; mbert-base: A 997, B 1,002. Bold: the two intervals do not overlap in that replicate.

### Task 2: ben source vs tel source, on targets hin+mal

A = ben→{hin,mal}; B = tel→{hin,mal}. Replicated over encoder: indicbert-v2, mbert-base.

| level | measure | A (indicbert-v2) | B (indicbert-v2) | A (mbert-base) | B (mbert-base) | verdict |
|---|---|---|---|---|---|---|
| L1 | linguistic | 7.3% [5.5%–9.3%] | 7.7% [6.1%–9.7%] | 9.5% [6.6%–12.7%] | 6.7% [4.5%–8.9%] | within noise |
| L1 | data | **53.2% [50.2%–56.2%]** | **21.7% [19.7%–23.8%]** | 53.3% [48.9%–57.3%] | 53.1% [49.2%–57.0%] | one encoder only |
| L1 | model | **21.7% [19.7%–24.1%]** | **65.8% [63.3%–68.2%]** | 20.9% [17.6%–24.5%] | 26.6% [23.2%–30.1%] | one encoder only |
| L1 | unattributed | **17.7% [15.5%–20.3%]** | **4.9% [4.0%–5.8%]** | 16.3% [13.0%–19.6%] | 13.6% [11.0%–16.7%] | one encoder only |
| L2 | translation_drift | 4.3% [3.0%–5.8%] | 4.1% [2.8%–5.6%] | 6.3% [3.7%–9.1%] | 5.2% [3.2%–7.3%] | within noise |
| L2 | tokenizer_fragmentation | **0.4% [0.1%–0.9%]** | **0.0% [0.0%–0.0%]** | 0.9% [0.0%–2.2%] | 0.3% [0.0%–0.9%] | one encoder only |
| L2 | morphological_masking | 0.2% [0.0%–0.7%] | 0.5% [0.1%–1.0%] | 0.0% [0.0%–0.0%] | 0.4% [0.0%–1.2%] | within noise |
| L2 | terminology_gap | 2.4% [1.4%–3.3%] | 3.0% [2.2%–4.1%] | 2.3% [1.2%–3.7%] | 0.7% [0.2%–1.3%] | within noise |
| L2 | provenance_specific | **22.0% [19.6%–24.5%]** | **14.3% [12.5%–16.2%]** | 26.2% [22.5%–30.1%] | 23.3% [20.2%–26.4%] | one encoder only |
| L2 | cross_lingual_hard | **31.1% [28.2%–34.1%]** | **7.3% [6.0%–8.7%]** | 27.1% [22.8%–31.0%] | 29.8% [26.2%–33.4%] | one encoder only |
| L2 | class_collapse | **8.6% [7.4%–9.8%]** | **63.7% [61.1%–66.2%]** | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | one encoder only |
| L2 | seed_unstable | **13.2% [11.4%–15.0%]** | **2.0% [1.4%–2.7%]** | 20.9% [17.6%–24.5%] | 26.6% [23.2%–30.1%] | one encoder only |

Failures counted — indicbert-v2: A 1,692, B 1,910; mbert-base: A 809, B 916. Bold: the two intervals do not overlap in that replicate.

### Task 3: hin source vs ben source, on targets mal+tel

A = hin→{mal,tel}; B = ben→{mal,tel}. Replicated over encoder: indicbert-v2, mbert-base.

| level | measure | A (indicbert-v2) | B (indicbert-v2) | A (mbert-base) | B (mbert-base) | verdict |
|---|---|---|---|---|---|---|
| L1 | linguistic | 12.3% [9.3%–15.4%] | 14.7% [10.9%–18.2%] | 14.8% [11.5%–18.3%] | 15.9% [12.5%–19.6%] | within noise |
| L1 | data | 73.1% [69.7%–76.3%] | 70.9% [67.0%–74.8%] | 62.4% [58.7%–66.4%] | 58.7% [54.6%–62.9%] | within noise |
| L1 | model | 3.6% [2.4%–4.9%] | 5.5% [4.0%–7.4%] | 10.0% [7.8%–12.4%] | 10.4% [8.1%–12.9%] | within noise |
| L1 | unattributed | 11.1% [8.9%–13.1%] | 8.9% [6.7%–11.4%] | 12.8% [10.3%–15.3%] | 15.0% [12.0%–18.1%] | within noise |
| L2 | translation_drift | 7.4% [5.0%–10.0%] | 11.8% [8.5%–15.0%] | 8.3% [5.8%–11.2%] | 13.1% [9.8%–16.6%] | within noise |
| L2 | tokenizer_fragmentation | 0.1% [0.0%–0.3%] | 0.3% [0.0%–0.8%] | **1.8% [0.7%–3.3%]** | **0.2% [0.0%–0.6%]** | one encoder only |
| L2 | morphological_masking | 2.8% [1.3%–4.5%] | 2.3% [0.9%–4.0%] | 2.9% [1.3%–4.6%] | 2.2% [0.8%–4.2%] | within noise |
| L2 | terminology_gap | **2.0% [1.1%–2.8%]** | **0.3% [0.0%–0.8%]** | 1.8% [0.8%–3.0%] | 0.4% [0.0%–0.8%] | one encoder only |
| L2 | provenance_specific | 7.6% [5.8%–9.5%] | 8.1% [5.9%–10.3%] | 15.4% [12.7%–18.2%] | 15.4% [12.8%–18.4%] | within noise |
| L2 | cross_lingual_hard | 65.5% [61.9%–69.0%] | 62.8% [58.8%–66.9%] | 47.0% [43.3%–50.9%] | 43.3% [39.3%–47.5%] | within noise |
| L2 | class_collapse | **0.0% [0.0%–0.0%]** | **2.1% [1.2%–3.3%]** | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | one encoder only |
| L2 | seed_unstable | 3.6% [2.4%–4.9%] | 3.4% [2.0%–4.8%] | 10.0% [7.8%–12.4%] | 10.4% [8.1%–12.9%] | within noise |

Failures counted — indicbert-v2: A 922, B 654; mbert-base: A 830, B 829. Bold: the two intervals do not overlap in that replicate.

### Task 3: hin source vs tel source, on targets ben+mal

A = hin→{ben,mal}; B = tel→{ben,mal}. Replicated over encoder: indicbert-v2, mbert-base.

| level | measure | A (indicbert-v2) | B (indicbert-v2) | A (mbert-base) | B (mbert-base) | verdict |
|---|---|---|---|---|---|---|
| L1 | linguistic | 13.0% [10.2%–16.1%] | 15.3% [11.3%–19.4%] | 15.0% [11.8%–18.5%] | 15.8% [11.9%–19.5%] | within noise |
| L1 | data | **71.5% [67.9%–74.7%]** | **54.9% [50.0%–59.9%]** | 62.8% [59.0%–66.9%] | 60.8% [56.7%–64.8%] | one encoder only |
| L1 | model | **3.8% [2.6%–5.1%]** | **13.7% [10.0%–17.5%]** | 9.5% [7.3%–11.9%] | 10.0% [7.8%–12.4%] | one encoder only |
| L1 | unattributed | 11.7% [9.5%–14.0%] | 16.2% [12.4%–20.0%] | 12.6% [10.0%–15.5%] | 13.4% [10.6%–16.4%] | within noise |
| L2 | translation_drift | 7.6% [5.4%–10.5%] | 13.7% [9.7%–17.9%] | 8.9% [6.3%–11.9%] | 14.3% [10.7%–18.0%] | within noise |
| L2 | tokenizer_fragmentation | 0.1% [0.0%–0.3%] | 0.0% [0.0%–0.0%] | **2.1% [0.9%–3.8%]** | **0.0% [0.0%–0.0%]** | one encoder only |
| L2 | morphological_masking | 1.8% [0.6%–3.3%] | 1.6% [0.2%–3.3%] | 1.6% [0.5%–3.1%] | 1.2% [0.2%–2.4%] | within noise |
| L2 | terminology_gap | **3.4% [2.3%–4.7%]** | **0.0% [0.0%–0.0%]** | **2.4% [1.2%–3.7%]** | **0.2% [0.0%–0.6%]** | held |
| L2 | provenance_specific | 7.9% [6.1%–9.7%] | 10.2% [7.4%–13.2%] | 15.5% [12.8%–18.1%] | 17.2% [14.4%–20.5%] | within noise |
| L2 | cross_lingual_hard | **63.6% [60.1%–66.9%]** | **44.7% [39.9%–49.3%]** | 47.3% [43.7%–51.4%] | 43.6% [39.6%–47.6%] | one encoder only |
| L2 | class_collapse | **0.0% [0.0%–0.0%]** | **2.1% [0.7%–3.5%]** | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | one encoder only |
| L2 | seed_unstable | **3.8% [2.6%–5.1%]** | **11.6% [8.0%–15.2%]** | 9.5% [7.3%–11.9%] | 10.0% [7.8%–12.4%] | one encoder only |

Failures counted — indicbert-v2: A 929, B 432; mbert-base: A 807, B 830. Bold: the two intervals do not overlap in that replicate.

### Task 3: ben source vs tel source, on targets hin+mal

A = ben→{hin,mal}; B = tel→{hin,mal}. Replicated over encoder: indicbert-v2, mbert-base.

| level | measure | A (indicbert-v2) | B (indicbert-v2) | A (mbert-base) | B (mbert-base) | verdict |
|---|---|---|---|---|---|---|
| L1 | linguistic | 9.6% [6.8%–12.6%] | 9.9% [6.4%–13.2%] | 11.7% [8.6%–14.8%] | 11.3% [8.3%–14.7%] | within noise |
| L1 | data | **72.9% [69.1%–76.5%]** | **61.3% [56.6%–66.2%]** | 61.9% [57.8%–66.3%] | 64.8% [60.6%–68.8%] | one encoder only |
| L1 | model | 9.4% [7.2%–11.9%] | 12.7% [9.5%–16.4%] | 12.2% [9.8%–15.0%] | 9.5% [7.4%–11.6%] | within noise |
| L1 | unattributed | **8.1% [6.1%–10.3%]** | **16.1% [12.2%–19.6%]** | 14.2% [11.4%–17.2%] | 14.4% [11.3%–17.6%] | one encoder only |
| L2 | translation_drift | 8.5% [5.9%–11.4%] | 9.9% [6.4%–13.2%] | 10.6% [7.5%–13.7%] | 10.8% [7.9%–14.2%] | within noise |
| L2 | tokenizer_fragmentation | 0.3% [0.0%–0.8%] | 0.0% [0.0%–0.0%] | 0.3% [0.0%–0.6%] | 0.0% [0.0%–0.0%] | within noise |
| L2 | morphological_masking | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | within noise |
| L2 | terminology_gap | **0.7% [0.1%–1.6%]** | **0.0% [0.0%–0.0%]** | 0.9% [0.3%–1.6%] | 0.5% [0.0%–1.2%] | one encoder only |
| L2 | provenance_specific | **8.4% [6.4%–10.5%]** | **13.9% [10.7%–17.2%]** | 13.8% [11.4%–16.8%] | 17.4% [14.7%–20.6%] | one encoder only |
| L2 | cross_lingual_hard | **64.6% [60.6%–68.3%]** | **47.4% [43.0%–52.2%]** | 48.1% [44.1%–52.4%] | 47.3% [43.4%–51.4%] | one encoder only |
| L2 | class_collapse | **7.0% [4.9%–9.2%]** | **2.2% [0.8%–3.7%]** | 0.0% [0.0%–0.0%] | 0.0% [0.0%–0.0%] | one encoder only |
| L2 | seed_unstable | **2.4% [1.2%–3.6%]** | **10.6% [7.6%–13.8%]** | 12.2% [9.8%–15.0%] | 9.5% [7.4%–11.6%] | one encoder only |

Failures counted — indicbert-v2: A 669, B 416; mbert-base: A 795, B 803. Bold: the two intervals do not overlap in that replicate.

## Without the runs that never trained (CLAUDE7.md hard rule 6)

3 comparison(s) include task-3 IndicBERT-v2 seeds that never trained and were re-run without them.
 These verdicts change:

| comparison | level | measure | all seeds | converged only |
|---|---|---|---|---|
| Task 3: hin source vs ben source, on targets mal+tel | L2 | class_collapse | one encoder only | within noise |
| Task 3: hin source vs tel source, on targets ben+mal | L1 | model | one encoder only | within noise |
| Task 3: hin source vs tel source, on targets ben+mal | L2 | class_collapse | one encoder only | within noise |
| Task 3: hin source vs tel source, on targets ben+mal | L2 | seed_unstable | one encoder only | within noise |
| Task 3: ben source vs tel source, on targets hin+mal | L1 | data | one encoder only | within noise |
| Task 3: ben source vs tel source, on targets hin+mal | L2 | provenance_specific | one encoder only | within noise |
| Task 3: ben source vs tel source, on targets hin+mal | L2 | seed_unstable | one encoder only | within noise |

