# Phase 7 — robustness of every candidate finding (T-706)

Config hash `bc5b24734953a7a2`. One row per (comparison, measure) from T-702–T-705. A difference is `held` only if the 95% CIs separate in every replicate and point the same way (CLAUDE7.md hard rule 5). In the share columns `*` marks a replicate whose intervals separate.

## Verdicts per question

| question | held | encoders disagree | tasks disagree | one encoder only | one task only | within noise |
|---|---|---|---|---|---|---|
| T-702 | 10 | 6 | 0 | 31 | 0 | 25 |
| T-703 | 3 | 0 | 0 | 36 | 0 | 33 |
| T-704 | 4 | 0 | 2 | 0 | 3 | 3 |
| T-705 | 5 | 0 | 0 | 8 | 0 | 59 |

## Held

Split by whether a known measurement artefact could produce the difference by itself (`hindi_morphology`, `native_drift`, `bengali_digits`).

### Held, no artefact caveat

| id | comparison | level | measure | A − B (mean) | shares (A vs B) | caveats |
|---|---|---|---|---|---|---|
| R702-03 | Task 2: into Dravidian vs out of it (hin↔tel, ben↔tel) | L1 | model | -8.2 pts | indicbert-v2: 34.4% vs 40.2% *; mbert-base: 14.4% vs 24.9% * |  |
| R702-12 | Task 2: into Dravidian vs out of it (hin↔tel, ben↔tel) | L2 | seed_unstable | -10.6 pts | indicbert-v2: 1.3% vs 12.0% *; mbert-base: 14.4% vs 24.9% * |  |
| R702-15 | Task 2: hin→tel vs tel→hin | L1 | model | -23.1 pts | indicbert-v2: 30.0% vs 61.4% *; mbert-base: 15.1% vs 30.0% * |  |
| R702-24 | Task 2: hin→tel vs tel→hin | L2 | seed_unstable | -8.6 pts | indicbert-v2: 1.3% vs 3.5% *; mbert-base: 15.1% vs 30.0% * |  |
| R703-15 | Task 2: hin source vs tel source, on targets ben+mal | L1 | model | -17.8 pts | indicbert-v2: 18.4% vs 46.8% *; mbert-base: 14.7% vs 22.0% * |  |
| R703-24 | Task 2: hin source vs tel source, on targets ben+mal | L2 | seed_unstable | -6.6 pts | indicbert-v2: 3.6% vs 9.6% *; mbert-base: 14.7% vs 22.0% * |  |
| R703-56 | Task 3: hin source vs tel source, on targets ben+mal | L2 | terminology_gap | +2.8 pts | indicbert-v2: 3.4% vs 0.0% *; mbert-base: 2.4% vs 0.2% * | task3_ceiling |
| R704-09 | IndicBERT-v2 vs mBERT, all native-text conditions | L2 | provenance_specific | -8.7 pts | 2: 15.9% vs 25.5% *; 3: 8.7% vs 16.6% * |  |
| R704-10 | IndicBERT-v2 vs mBERT, all native-text conditions | L2 | cross_lingual_hard | +9.1 pts | 2: 28.5% vs 24.4% *; 3: 60.6% vs 46.6% * |  |
| R704-11 | IndicBERT-v2 vs mBERT, all native-text conditions | L2 | class_collapse | +14.8 pts | 2: 27.3% vs 0.0% *; 3: 2.3% vs 0.0% * |  |
| R704-12 | IndicBERT-v2 vs mBERT, all native-text conditions | L2 | seed_unstable | -9.1 pts | 2: 6.3% vs 19.0% *; 3: 5.0% vs 10.5% * |  |

### Held, but an artefact could produce it

| id | comparison | level | measure | A − B (mean) | shares (A vs B) | caveats |
|---|---|---|---|---|---|---|
| R702-01 | Task 2: into Dravidian vs out of it (hin↔tel, ben↔tel) | L1 | linguistic | +9.8 pts | indicbert-v2: 20.6% vs 11.4% *; mbert-base: 18.4% vs 8.0% * | hindi_morphology |
| R702-07 | Task 2: into Dravidian vs out of it (hin↔tel, ben↔tel) | L2 | morphological_masking | +11.4 pts | indicbert-v2: 19.6% vs 6.4% *; mbert-base: 17.3% vs 7.6% * | hindi_morphology |
| R702-13 | Task 2: hin→tel vs tel→hin | L1 | linguistic | +17.0 pts | indicbert-v2: 20.2% vs 4.8% *; mbert-base: 18.9% vs 0.2% * | hindi_morphology |
| R702-19 | Task 2: hin→tel vs tel→hin | L2 | morphological_masking | +18.3 pts | indicbert-v2: 19.1% vs 0.0% *; mbert-base: 17.5% vs 0.0% * | hindi_morphology |
| R702-43 | Task 3: into Dravidian vs out of it (hin↔tel, ben↔tel) | L2 | morphological_masking | +3.8 pts | indicbert-v2: 5.2% vs 1.6% *; mbert-base: 5.1% vs 1.2% * | hindi_morphology, task3_ceiling |
| R702-55 | Task 3: hin→tel vs tel→hin | L2 | morphological_masking | +5.7 pts | indicbert-v2: 5.6% vs 0.0% *; mbert-base: 5.9% vs 0.0% * | hindi_morphology, task3_ceiling |
| R705-17 | Task 2: hin target, native vs MT text | L2 | translation_drift | -0.4 pts | indicbert-v2: 0.0% vs 0.3% *; mbert-base: 0.0% vs 0.4% * | native_drift |
| R705-29 | Task 2: tel target, native vs MT text | L2 | translation_drift | -0.8 pts | indicbert-v2: 0.0% vs 0.6% *; mbert-base: 0.0% vs 0.9% * | native_drift |
| R705-37 | Task 3: ben target, native vs MT text | L1 | linguistic | +7.1 pts | indicbert-v2: 13.0% vs 6.6% *; mbert-base: 13.6% vs 5.8% * | native_drift, bengali_digits, task3_ceiling |
| R705-41 | Task 3: ben target, native vs MT text | L2 | translation_drift | +6.4 pts | indicbert-v2: 6.3% vs 1.3% *; mbert-base: 8.7% vs 0.9% * | native_drift, bengali_digits, task3_ceiling |
| R705-65 | Task 3: tel target, native vs MT text | L2 | translation_drift | +6.3 pts | indicbert-v2: 5.8% vs 0.6% *; mbert-base: 8.1% vs 0.8% * | native_drift, task3_ceiling |

## Replicates point opposite ways

| id | comparison | level | measure | A − B (mean) | shares (A vs B) | caveats |
|---|---|---|---|---|---|---|
| R702-02 | Task 2: into Dravidian vs out of it (hin↔tel, ben↔tel) | L1 | data | +2.4 pts | indicbert-v2: 44.3% vs 31.2% *; mbert-base: 43.2% vs 51.5% * |  |
| R702-04 | Task 2: into Dravidian vs out of it (hin↔tel, ben↔tel) | L1 | unattributed | -4.0 pts | indicbert-v2: 0.7% vs 17.2% *; mbert-base: 24.0% vs 15.5% * |  |
| R702-10 | Task 2: into Dravidian vs out of it (hin↔tel, ben↔tel) | L2 | cross_lingual_hard | +9.3 pts | indicbert-v2: 35.3% vs 7.5% *; mbert-base: 20.1% vs 29.4% * |  |
| R702-14 | Task 2: hin→tel vs tel→hin | L1 | data | +6.2 pts | indicbert-v2: 49.3% vs 23.2% *; mbert-base: 39.1% vs 52.9% * |  |
| R702-16 | Task 2: hin→tel vs tel→hin | L1 | unattributed | -0.1 pts | indicbert-v2: 0.5% vs 10.6% *; mbert-base: 26.9% vs 16.9% * |  |
| R702-22 | Task 2: hin→tel vs tel→hin | L2 | cross_lingual_hard | +9.5 pts | indicbert-v2: 43.2% vs 7.7% *; mbert-base: 18.7% vs 35.1% * |  |
| R704-02 | IndicBERT-v2 vs mBERT, all native-text conditions | L1 | data | +0.3 pts | 2: 44.5% vs 49.9% *; 3: 69.3% vs 63.2% * |  |
| R704-03 | IndicBERT-v2 vs mBERT, all native-text conditions | L1 | model | +5.7 pts | 2: 33.5% vs 19.0% *; 3: 7.4% vs 10.5% * |  |

## Separated in one replicate only

| id | comparison | level | measure | A − B (mean) | shares (A vs B) | caveats |
|---|---|---|---|---|---|---|
| R702-08 | Task 2: into Dravidian vs out of it (hin↔tel, ben↔tel) | L2 | terminology_gap | -1.9 pts | indicbert-v2: 0.9% vs 5.0% *; mbert-base: 0.6% vs 0.4% |  |
| R702-09 | Task 2: into Dravidian vs out of it (hin↔tel, ben↔tel) | L2 | provenance_specific | -6.9 pts | indicbert-v2: 8.9% vs 23.8% *; mbert-base: 23.2% vs 22.1% |  |
| R702-20 | Task 2: hin→tel vs tel→hin | L2 | terminology_gap | -1.8 pts | indicbert-v2: 0.9% vs 4.8% *; mbert-base: 0.6% vs 0.2% |  |
| R702-21 | Task 2: hin→tel vs tel→hin | L2 | provenance_specific | -3.4 pts | indicbert-v2: 6.2% vs 15.5% *; mbert-base: 20.4% vs 17.8% |  |
| R702-23 | Task 2: hin→tel vs tel→hin | L2 | class_collapse | -14.5 pts | indicbert-v2: 28.7% vs 57.8% *; mbert-base: 0.0% vs 0.0% |  |
| R702-27 | Task 2: ben→tel vs tel→ben | L1 | model | +5.9 pts | indicbert-v2: 39.2% vs 20.0% *; mbert-base: 13.4% vs 20.7% |  |
| R702-28 | Task 2: ben→tel vs tel→ben | L1 | unattributed | -8.3 pts | indicbert-v2: 1.0% vs 23.4% *; mbert-base: 20.2% vs 14.4% |  |
| R702-31 | Task 2: ben→tel vs tel→ben | L2 | morphological_masking | +5.4 pts | indicbert-v2: 20.1% vs 12.5% *; mbert-base: 17.2% vs 14.0% |  |
| R702-32 | Task 2: ben→tel vs tel→ben | L2 | terminology_gap | -2.1 pts | indicbert-v2: 0.9% vs 5.2% *; mbert-base: 0.6% vs 0.6% |  |
| R702-33 | Task 2: ben→tel vs tel→ben | L2 | provenance_specific | -9.3 pts | indicbert-v2: 11.9% vs 31.6% *; mbert-base: 26.8% vs 25.7% |  |
| R702-34 | Task 2: ben→tel vs tel→ben | L2 | cross_lingual_hard | +8.4 pts | indicbert-v2: 26.9% vs 7.3% *; mbert-base: 21.8% vs 24.6% |  |
| R702-35 | Task 2: ben→tel vs tel→ben | L2 | class_collapse | +18.9 pts | indicbert-v2: 37.9% vs 0.0% *; mbert-base: 0.0% vs 0.0% |  |
| R702-36 | Task 2: ben→tel vs tel→ben | L2 | seed_unstable | -13.0 pts | indicbert-v2: 1.4% vs 20.0% *; mbert-base: 13.4% vs 20.7% |  |
| R702-38 | Task 3: into Dravidian vs out of it (hin↔tel, ben↔tel) | L1 | data | +7.0 pts | indicbert-v2: 73.0% vs 57.7% *; mbert-base: 64.2% vs 65.5% | task3_ceiling |
| R702-39 | Task 3: into Dravidian vs out of it (hin↔tel, ben↔tel) | L1 | model | -5.6 pts | indicbert-v2: 4.7% vs 14.5% *; mbert-base: 10.0% vs 11.5% | task3_ceiling |
| R702-40 | Task 3: into Dravidian vs out of it (hin↔tel, ben↔tel) | L1 | unattributed | -4.7 pts | indicbert-v2: 10.2% vs 18.0% *; mbert-base: 11.4% vs 12.9% | task3_ceiling |
| R702-44 | Task 3: into Dravidian vs out of it (hin↔tel, ben↔tel) | L2 | terminology_gap | +0.8 pts | indicbert-v2: 1.1% vs 0.0% *; mbert-base: 1.1% vs 0.7% | task3_ceiling |
| R702-45 | Task 3: into Dravidian vs out of it (hin↔tel, ben↔tel) | L2 | provenance_specific | -2.5 pts | indicbert-v2: 7.2% vs 12.7% *; mbert-base: 18.9% vs 18.4% | task3_ceiling |
| R702-46 | Task 3: into Dravidian vs out of it (hin↔tel, ben↔tel) | L2 | cross_lingual_hard | +9.5 pts | indicbert-v2: 65.7% vs 45.0% *; mbert-base: 45.3% vs 47.1% | task3_ceiling |
| R702-47 | Task 3: into Dravidian vs out of it (hin↔tel, ben↔tel) | L2 | class_collapse | +0.9 pts | indicbert-v2: 1.8% vs 0.0% *; mbert-base: 0.0% vs 0.0% | task3_ceiling |
| R702-48 | Task 3: into Dravidian vs out of it (hin↔tel, ben↔tel) | L2 | seed_unstable | -6.5 pts | indicbert-v2: 2.9% vs 14.5% *; mbert-base: 10.0% vs 11.5% | task3_ceiling |
| R702-51 | Task 3: hin→tel vs tel→hin | L1 | model | -5.8 pts | indicbert-v2: 3.2% vs 13.7% *; mbert-base: 9.9% vs 11.0% | task3_ceiling |
| R702-56 | Task 3: hin→tel vs tel→hin | L2 | terminology_gap | +1.4 pts | indicbert-v2: 1.9% vs 0.0% *; mbert-base: 2.0% vs 1.0% | task3_ceiling |
| R702-57 | Task 3: hin→tel vs tel→hin | L2 | provenance_specific | -3.7 pts | indicbert-v2: 7.9% vs 16.5% *; mbert-base: 19.7% vs 18.6% | task3_ceiling |
| R702-58 | Task 3: hin→tel vs tel→hin | L2 | cross_lingual_hard | +7.8 pts | indicbert-v2: 66.7% vs 47.6% *; mbert-base: 47.5% vs 51.0% | task3_ceiling |
| R702-60 | Task 3: hin→tel vs tel→hin | L2 | seed_unstable | -5.8 pts | indicbert-v2: 3.2% vs 13.7% *; mbert-base: 9.9% vs 11.0% | task3_ceiling |
| R702-62 | Task 3: ben→tel vs tel→ben | L1 | data | +9.2 pts | indicbert-v2: 70.5% vs 51.8% *; mbert-base: 61.2% vs 61.6% | task3_ceiling |
| R702-64 | Task 3: ben→tel vs tel→ben | L1 | unattributed | -4.8 pts | indicbert-v2: 8.1% vs 18.0% *; mbert-base: 12.2% vs 11.9% | task3_ceiling |
| R702-70 | Task 3: ben→tel vs tel→ben | L2 | cross_lingual_hard | +10.7 pts | indicbert-v2: 64.3% vs 42.5% *; mbert-base: 43.2% vs 43.4% | task3_ceiling |
| R702-71 | Task 3: ben→tel vs tel→ben | L2 | class_collapse | +2.2 pts | indicbert-v2: 4.3% vs 0.0% *; mbert-base: 0.0% vs 0.0% | task3_ceiling |
| R702-72 | Task 3: ben→tel vs tel→ben | L2 | seed_unstable | -7.3 pts | indicbert-v2: 2.5% vs 15.4% *; mbert-base: 10.2% vs 11.9% | task3_ceiling |
| R703-02 | Task 2: hin source vs ben source, on targets mal+tel | L1 | data | +4.4 pts | indicbert-v2: 56.6% vs 42.5% *; mbert-base: 40.4% vs 45.7% |  |
| R703-03 | Task 2: hin source vs ben source, on targets mal+tel | L1 | model | -5.6 pts | indicbert-v2: 23.7% vs 33.1% *; mbert-base: 16.3% vs 18.1% |  |
| R703-04 | Task 2: hin source vs ben source, on targets mal+tel | L1 | unattributed | +0.6 pts | indicbert-v2: 3.5% vs 8.7% *; mbert-base: 27.0% vs 20.6% |  |
| R703-09 | Task 2: hin source vs ben source, on targets mal+tel | L2 | provenance_specific | -3.7 pts | indicbert-v2: 10.0% vs 15.1% *; mbert-base: 21.7% vs 24.0% |  |
| R703-10 | Task 2: hin source vs ben source, on targets mal+tel | L2 | cross_lingual_hard | +8.1 pts | indicbert-v2: 46.6% vs 27.4% *; mbert-base: 18.6% vs 21.7% |  |
| R703-11 | Task 2: hin source vs ben source, on targets mal+tel | L2 | class_collapse | -3.2 pts | indicbert-v2: 20.5% vs 26.8% *; mbert-base: 0.0% vs 0.0% |  |
| R703-12 | Task 2: hin source vs ben source, on targets mal+tel | L2 | seed_unstable | -2.5 pts | indicbert-v2: 3.2% vs 6.3% *; mbert-base: 16.3% vs 18.1% |  |
| R703-14 | Task 2: hin source vs tel source, on targets ben+mal | L1 | data | +16.0 pts | indicbert-v2: 60.9% vs 28.9% *; mbert-base: 51.6% vs 51.7% |  |
| R703-16 | Task 2: hin source vs tel source, on targets ben+mal | L1 | unattributed | +3.0 pts | indicbert-v2: 8.0% vs 10.8%; mbert-base: 21.4% vs 12.6% * |  |
| R703-21 | Task 2: hin source vs tel source, on targets ben+mal | L2 | provenance_specific | -3.2 pts | indicbert-v2: 12.9% vs 21.7% *; mbert-base: 29.5% vs 27.0% |  |
| R703-22 | Task 2: hin source vs tel source, on targets ben+mal | L2 | cross_lingual_hard | +19.2 pts | indicbert-v2: 48.1% vs 7.2% *; mbert-base: 22.1% vs 24.7% |  |
| R703-23 | Task 2: hin source vs tel source, on targets ben+mal | L2 | class_collapse | -11.2 pts | indicbert-v2: 14.7% vs 37.2% *; mbert-base: 0.0% vs 0.0% |  |
| R703-26 | Task 2: ben source vs tel source, on targets hin+mal | L1 | data | +15.9 pts | indicbert-v2: 53.2% vs 21.7% *; mbert-base: 53.3% vs 53.1% |  |
| R703-27 | Task 2: ben source vs tel source, on targets hin+mal | L1 | model | -24.9 pts | indicbert-v2: 21.7% vs 65.8% *; mbert-base: 20.9% vs 26.6% |  |
| R703-28 | Task 2: ben source vs tel source, on targets hin+mal | L1 | unattributed | +7.8 pts | indicbert-v2: 17.7% vs 4.9% *; mbert-base: 16.3% vs 13.6% |  |
| R703-30 | Task 2: ben source vs tel source, on targets hin+mal | L2 | tokenizer_fragmentation | +0.5 pts | indicbert-v2: 0.4% vs 0.0% *; mbert-base: 0.9% vs 0.3% |  |
| R703-33 | Task 2: ben source vs tel source, on targets hin+mal | L2 | provenance_specific | +5.3 pts | indicbert-v2: 22.0% vs 14.3% *; mbert-base: 26.2% vs 23.3% |  |
| R703-34 | Task 2: ben source vs tel source, on targets hin+mal | L2 | cross_lingual_hard | +10.5 pts | indicbert-v2: 31.1% vs 7.3% *; mbert-base: 27.1% vs 29.8% |  |
| R703-35 | Task 2: ben source vs tel source, on targets hin+mal | L2 | class_collapse | -27.6 pts | indicbert-v2: 8.6% vs 63.7% *; mbert-base: 0.0% vs 0.0% |  |
| R703-36 | Task 2: ben source vs tel source, on targets hin+mal | L2 | seed_unstable | +2.7 pts | indicbert-v2: 13.2% vs 2.0% *; mbert-base: 20.9% vs 26.6% |  |
| R703-42 | Task 3: hin source vs ben source, on targets mal+tel | L2 | tokenizer_fragmentation | +0.7 pts | indicbert-v2: 0.1% vs 0.3%; mbert-base: 1.8% vs 0.2% * | task3_ceiling |
| R703-44 | Task 3: hin source vs ben source, on targets mal+tel | L2 | terminology_gap | +1.5 pts | indicbert-v2: 2.0% vs 0.3% *; mbert-base: 1.8% vs 0.4% | task3_ceiling |
| R703-47 | Task 3: hin source vs ben source, on targets mal+tel | L2 | class_collapse | -1.1 pts | indicbert-v2: 0.0% vs 2.1% *; mbert-base: 0.0% vs 0.0% | task3_ceiling |
| R703-50 | Task 3: hin source vs tel source, on targets ben+mal | L1 | data | +9.3 pts | indicbert-v2: 71.5% vs 54.9% *; mbert-base: 62.8% vs 60.8% | task3_ceiling |
| R703-51 | Task 3: hin source vs tel source, on targets ben+mal | L1 | model | -5.2 pts | indicbert-v2: 3.8% vs 13.7% *; mbert-base: 9.5% vs 10.0% | task3_ceiling |
| R703-54 | Task 3: hin source vs tel source, on targets ben+mal | L2 | tokenizer_fragmentation | +1.1 pts | indicbert-v2: 0.1% vs 0.0%; mbert-base: 2.1% vs 0.0% * | task3_ceiling |
| R703-58 | Task 3: hin source vs tel source, on targets ben+mal | L2 | cross_lingual_hard | +11.3 pts | indicbert-v2: 63.6% vs 44.7% *; mbert-base: 47.3% vs 43.6% | task3_ceiling |
| R703-59 | Task 3: hin source vs tel source, on targets ben+mal | L2 | class_collapse | -1.0 pts | indicbert-v2: 0.0% vs 2.1% *; mbert-base: 0.0% vs 0.0% | task3_ceiling |
| R703-60 | Task 3: hin source vs tel source, on targets ben+mal | L2 | seed_unstable | -4.1 pts | indicbert-v2: 3.8% vs 11.6% *; mbert-base: 9.5% vs 10.0% | task3_ceiling |
| R703-62 | Task 3: ben source vs tel source, on targets hin+mal | L1 | data | +4.4 pts | indicbert-v2: 72.9% vs 61.3% *; mbert-base: 61.9% vs 64.8% | task3_ceiling |
| R703-64 | Task 3: ben source vs tel source, on targets hin+mal | L1 | unattributed | -4.1 pts | indicbert-v2: 8.1% vs 16.1% *; mbert-base: 14.2% vs 14.4% | task3_ceiling |
| R703-68 | Task 3: ben source vs tel source, on targets hin+mal | L2 | terminology_gap | +0.6 pts | indicbert-v2: 0.7% vs 0.0% *; mbert-base: 0.9% vs 0.5% | task3_ceiling |
| R703-69 | Task 3: ben source vs tel source, on targets hin+mal | L2 | provenance_specific | -4.6 pts | indicbert-v2: 8.4% vs 13.9% *; mbert-base: 13.8% vs 17.4% | task3_ceiling |
| R703-70 | Task 3: ben source vs tel source, on targets hin+mal | L2 | cross_lingual_hard | +9.0 pts | indicbert-v2: 64.6% vs 47.4% *; mbert-base: 48.1% vs 47.3% | task3_ceiling |
| R703-71 | Task 3: ben source vs tel source, on targets hin+mal | L2 | class_collapse | +2.4 pts | indicbert-v2: 7.0% vs 2.2% *; mbert-base: 0.0% vs 0.0% | task3_ceiling |
| R703-72 | Task 3: ben source vs tel source, on targets hin+mal | L2 | seed_unstable | -2.7 pts | indicbert-v2: 2.4% vs 10.6% *; mbert-base: 12.2% vs 9.5% | task3_ceiling |
| R704-04 | IndicBERT-v2 vs mBERT, all native-text conditions | L1 | unattributed | -5.5 pts | 2: 9.0% vs 18.7% *; 3: 11.6% vs 13.0% |  |
| R704-06 | IndicBERT-v2 vs mBERT, all native-text conditions | L2 | tokenizer_fragmentation | -0.5 pts | 2: 0.1% vs 0.6% *; 3: 0.1% vs 0.5% |  |
| R704-08 | IndicBERT-v2 vs mBERT, all native-text conditions | L2 | terminology_gap | +0.9 pts | 2: 2.3% vs 1.0% *; 3: 1.5% vs 1.1% |  |
| R705-03 | Task 2: ben target, native vs MT text | L1 | model | -4.4 pts | indicbert-v2: 20.0% vs 22.9%; mbert-base: 16.4% vs 22.3% * | bengali_digits |
| R705-05 | Task 2: ben target, native vs MT text | L2 | translation_drift | -0.1 pts | indicbert-v2: 0.0% vs 0.2% *; mbert-base: 0.0% vs 0.0% | native_drift, bengali_digits |
| R705-08 | Task 2: ben target, native vs MT text | L2 | terminology_gap | -0.7 pts | indicbert-v2: 3.2% vs 3.2%; mbert-base: 0.3% vs 1.8% * | bengali_digits |
| R705-09 | Task 2: ben target, native vs MT text | L2 | provenance_specific | +7.3 pts | indicbert-v2: 20.6% vs 17.2%; mbert-base: 31.3% vs 19.9% * | bengali_digits |
| R705-12 | Task 2: ben target, native vs MT text | L2 | seed_unstable | -4.0 pts | indicbert-v2: 10.2% vs 12.3%; mbert-base: 16.4% vs 22.3% * | bengali_digits |
| R705-33 | Task 2: tel target, native vs MT text | L2 | provenance_specific | +4.2 pts | indicbert-v2: 8.9% vs 9.1%; mbert-base: 23.2% vs 14.6% * |  |
| R705-53 | Task 3: hin target, native vs MT text | L2 | translation_drift | +3.6 pts | indicbert-v2: 3.8% vs 0.9%; mbert-base: 5.2% vs 0.9% * | native_drift, task3_ceiling |
| R705-71 | Task 3: tel target, native vs MT text | L2 | class_collapse | +0.9 pts | indicbert-v2: 1.8% vs 0.0% *; mbert-base: 0.0% vs 0.0% | task3_ceiling |

## Within noise

120 rows; see the parquet.

## Without the runs that never trained (hard rule 6)

120 rows were re-run without them; these verdicts change:

| id | comparison | measure | all seeds | converged only | shares, converged only |
|---|---|---|---|---|---|
| R702-40 | Task 3: into Dravidian vs out of it (hin↔tel, ben↔tel) | unattributed | one encoder only | within noise | indicbert-v2: 10.4% vs 14.9%; mbert-base: 11.4% vs 12.9% |
| R702-45 | Task 3: into Dravidian vs out of it (hin↔tel, ben↔tel) | provenance_specific | one encoder only | within noise | indicbert-v2: 7.2% vs 9.9%; mbert-base: 18.9% vs 18.4% |
| R702-47 | Task 3: into Dravidian vs out of it (hin↔tel, ben↔tel) | class_collapse | one encoder only | within noise | indicbert-v2: 0.0% vs 0.0%; mbert-base: 0.0% vs 0.0% |
| R702-49 | Task 3: hin→tel vs tel→hin | linguistic | within noise | one encoder only | indicbert-v2: 10.5% vs 4.1% *; mbert-base: 12.3% vs 5.4% |
| R702-51 | Task 3: hin→tel vs tel→hin | model | one encoder only | within noise | indicbert-v2: 3.2% vs 8.2%; mbert-base: 9.9% vs 11.0% |
| R702-57 | Task 3: hin→tel vs tel→hin | provenance_specific | one encoder only | within noise | indicbert-v2: 7.9% vs 14.4%; mbert-base: 19.7% vs 18.6% |
| R702-58 | Task 3: hin→tel vs tel→hin | cross_lingual_hard | one encoder only | within noise | indicbert-v2: 66.7% vs 58.2%; mbert-base: 47.5% vs 51.0% |
| R702-60 | Task 3: hin→tel vs tel→hin | seed_unstable | one encoder only | within noise | indicbert-v2: 3.2% vs 8.2%; mbert-base: 9.9% vs 11.0% |
| R702-63 | Task 3: ben→tel vs tel→ben | model | within noise | one encoder only | indicbert-v2: 2.7% vs 12.8% *; mbert-base: 10.2% vs 11.9% |
| R702-64 | Task 3: ben→tel vs tel→ben | unattributed | one encoder only | within noise | indicbert-v2: 8.7% vs 14.7%; mbert-base: 12.2% vs 11.9% |
| R702-71 | Task 3: ben→tel vs tel→ben | class_collapse | one encoder only | within noise | indicbert-v2: 0.0% vs 0.0%; mbert-base: 0.0% vs 0.0% |
| R703-47 | Task 3: hin source vs ben source, on targets mal+tel | class_collapse | one encoder only | within noise | indicbert-v2: 0.0% vs 0.0%; mbert-base: 0.0% vs 0.0% |
| R703-51 | Task 3: hin source vs tel source, on targets ben+mal | model | one encoder only | within noise | indicbert-v2: 3.8% vs 7.9%; mbert-base: 9.5% vs 10.0% |
| R703-59 | Task 3: hin source vs tel source, on targets ben+mal | class_collapse | one encoder only | within noise | indicbert-v2: 0.0% vs 0.0%; mbert-base: 0.0% vs 0.0% |
| R703-60 | Task 3: hin source vs tel source, on targets ben+mal | seed_unstable | one encoder only | within noise | indicbert-v2: 3.8% vs 7.9%; mbert-base: 9.5% vs 10.0% |
| R703-62 | Task 3: ben source vs tel source, on targets hin+mal | data | one encoder only | within noise | indicbert-v2: 76.2% vs 68.2%; mbert-base: 61.9% vs 64.8% |
| R703-69 | Task 3: ben source vs tel source, on targets hin+mal | provenance_specific | one encoder only | within noise | indicbert-v2: 7.5% vs 11.3%; mbert-base: 13.8% vs 17.4% |
| R703-72 | Task 3: ben source vs tel source, on targets hin+mal | seed_unstable | one encoder only | within noise | indicbert-v2: 2.4% vs 5.5%; mbert-base: 12.2% vs 9.5% |
| R705-53 | Task 3: hin target, native vs MT text | translation_drift | one encoder only | held | indicbert-v2: 3.7% vs 0.7% *; mbert-base: 5.2% vs 0.9% * |
| R705-71 | Task 3: tel target, native vs MT text | class_collapse | one encoder only | within noise | indicbert-v2: 0.0% vs 0.0%; mbert-base: 0.0% vs 0.0% |

## Caveat key

- `hindi_morphology`: one side includes Hindi targets, which cannot receive morphological_masking (Hindi writes postpositions as separate words); the difference may be that blind spot.
- `native_drift`: translation_drift on native text means the *human* translation scored below tau; tau passes 95% of human pairs by construction (T-110) and LaBSE scores literal MT above human translation, so native-vs-MT drift differences reflect the measure, not translation damage.
- `bengali_digits`: native and MT Bengali also differ in digit system (CLAUDE4.md, Scope decisions).
- `task3_ceiling`: task 3 is at its data ceiling (Phase 4).
