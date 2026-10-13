# Phase 7 — cause mix by encoder (T-704)

Config hash `bc5b24734953a7a2`. Shares are of failed (item, seed) rows (CLAUDE7.md). A difference is a finding only if the two 95% CIs do not overlap and the direction agrees in every replicate (hard rule 5).

Two labels carry a stated limitation (hard rule 4):

- `morphological_masking`: a masked term does not raise the failure rate (masked fails more often than intact in 20 of 44 cells; reports/morphology_null.md).
- `terminology_gap`: salience divergence does not separate failures from correct pairs (AUC 0.48-0.52; reports/saliency_baseline.md).

**Scope.** Native-text conditions only, all nine source→target pairs pooled per encoder. Both encoders are scored on the same items and conditions. mBERT fails on fewer rows, so a share here is a share of a smaller set; `fail rate` gives the volume. The replicate is the task, and task 3 is at its data ceiling (Phase 4), so a difference must survive a task with little signal.

## Cause mix by encoder

| task | encoder | failures | fail rate | unattributed | linguistic | data | model | translation_drift | tokenizer_fragmentation | morphological_masking | terminology_gap | provenance_specific | cross_lingual_hard | class_collapse | seed_unstable |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 2 | indicbert-v2 | 8,703 | 29.6% | 9.0% [8.3%–9.7%] | 13.0% [12.0%–14.0%] | 44.5% [43.2%–45.8%] | 33.5% [32.2%–34.8%] | 2.7% | 0.1% | 7.8% | 2.3% | 15.9% | 28.5% | 27.3% | 6.3% |
| 2 | mbert-base | 4,405 | 15.0% | 18.7% [17.1%–20.3%] | 12.5% [11.1%–13.9%] | 49.9% [47.8%–51.7%] | 19.0% [17.6%–20.6%] | 3.5% | 0.6% | 7.4% | 1.0% | 25.5% | 24.4% | 0.0% | 19.0% |
| 3 | indicbert-v2 | 3,030 | 41.0% | 11.6% [10.4%–12.8%] | 11.7% [10.3%–13.3%] | 69.3% [67.4%–71.1%] | 7.4% [6.3%–8.4%] | 8.0% | 0.1% | 2.1% | 1.5% | 8.7% | 60.6% | 2.3% | 5.0% |
| 3 | mbert-base | 3,640 | 49.2% | 13.0% [11.7%–14.2%] | 13.3% [11.7%–15.0%] | 63.2% [61.3%–65.1%] | 10.5% [9.4%–11.5%] | 9.8% | 0.5% | 1.8% | 1.1% | 16.6% | 46.6% | 0.0% | 10.5% |

## Comparisons

### IndicBERT-v2 vs mBERT, all native-text conditions

A = indicbert-v2; B = mbert-base. Replicated over task: 2, 3.

| level | measure | A (2) | B (2) | A (3) | B (3) | verdict |
|---|---|---|---|---|---|---|
| L1 | linguistic | 13.0% [12.0%–14.0%] | 12.5% [11.1%–13.9%] | 11.7% [10.3%–13.3%] | 13.3% [11.7%–15.0%] | within noise |
| L1 | data | **44.5% [43.2%–45.8%]** | **49.9% [47.8%–51.7%]** | **69.3% [67.4%–71.1%]** | **63.2% [61.3%–65.1%]** | tasks disagree |
| L1 | model | **33.5% [32.2%–34.8%]** | **19.0% [17.6%–20.6%]** | **7.4% [6.3%–8.4%]** | **10.5% [9.4%–11.5%]** | tasks disagree |
| L1 | unattributed | **9.0% [8.3%–9.7%]** | **18.7% [17.1%–20.3%]** | 11.6% [10.4%–12.8%] | 13.0% [11.7%–14.2%] | one task only |
| L2 | translation_drift | 2.7% [2.2%–3.3%] | 3.5% [2.8%–4.4%] | 8.0% [6.7%–9.2%] | 9.8% [8.5%–11.3%] | within noise |
| L2 | tokenizer_fragmentation | **0.1% [0.0%–0.2%]** | **0.6% [0.3%–1.0%]** | 0.1% [0.0%–0.2%] | 0.5% [0.2%–0.9%] | one task only |
| L2 | morphological_masking | 7.8% [7.0%–8.8%] | 7.4% [6.4%–8.5%] | 2.1% [1.5%–3.0%] | 1.8% [1.1%–2.5%] | within noise |
| L2 | terminology_gap | **2.3% [2.0%–2.7%]** | **1.0% [0.6%–1.3%]** | 1.5% [1.1%–2.0%] | 1.1% [0.8%–1.5%] | one task only |
| L2 | provenance_specific | **15.9% [15.1%–16.9%]** | **25.5% [23.9%–27.1%]** | **8.7% [7.7%–9.8%]** | **16.6% [15.2%–17.9%]** | held |
| L2 | cross_lingual_hard | **28.5% [27.3%–29.9%]** | **24.4% [22.7%–25.9%]** | **60.6% [58.5%–62.5%]** | **46.6% [44.8%–48.6%]** | held |
| L2 | class_collapse | **27.3% [26.1%–28.6%]** | **0.0% [0.0%–0.0%]** | **2.3% [1.8%–2.9%]** | **0.0% [0.0%–0.0%]** | held |
| L2 | seed_unstable | **6.3% [5.7%–6.9%]** | **19.0% [17.6%–20.6%]** | **5.0% [4.2%–6.0%]** | **10.5% [9.4%–11.5%]** | held |

Failures counted — 2: A 8,703, B 4,405; 3: A 3,030, B 3,640. Bold: the two intervals do not overlap in that replicate.

## Without the runs that never trained (CLAUDE7.md hard rule 6)

1 comparison(s) include task-3 IndicBERT-v2 seeds that never trained and were re-run without them.
 No verdict changes.

