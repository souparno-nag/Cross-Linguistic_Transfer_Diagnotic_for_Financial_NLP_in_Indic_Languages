# T-110 — calibrating τ, task 2

τ in effect = 0.82. **Verdict: confirmed.**

## What MT scores

| src_lang   | tgt_lang   |   rows |   median |    p05 |   below_share |
|:-----------|:-----------|-------:|---------:|-------:|--------------:|
| ben        | mal        |   2228 |   0.9016 | 0.8174 |        0.0543 |
| hin        | mal        |   2238 |   0.9075 | 0.8237 |        0.0442 |
| ben        | tel        |   2228 |   0.9229 | 0.8447 |        0.0238 |
| tel        | mal        |   2072 |   0.924  | 0.8476 |        0.0159 |
| ben        | hin        |   2228 |   0.9309 | 0.8599 |        0.0076 |
| tel        | ben        |   2072 |   0.9389 | 0.8758 |        0.0068 |
| tel        | hin        |   2072 |   0.9432 | 0.8847 |        0.0039 |
| hin        | ben        |   2238 |   0.9495 | 0.895  |        0.0031 |
| hin        | tel        |   2238 |   0.9445 | 0.8813 |        0.0022 |

## What a human translation scores, on the same measure

The native splits of this task are human translations of one another (§2.1), so the cosine between two of them for the same item is a known-good pair measured exactly like an MT pair. This is the ceiling.

| pair    |   rows |   median |    p05 |   below_tau |
|:--------|-------:|---------:|-------:|------------:|
| ben-hin |   1769 |   0.9178 | 0.8468 |           0 |
| ben-tel |   1769 |   0.9054 | 0.8362 |           0 |
| hin-tel |   1769 |   0.9328 | 0.8737 |           0 |

## MT against a human translation of the same item

Same language, same item, so this isolates translation quality from cross-language drift. No Malayalam row appears: there is no native Malayalam anywhere in IndicFinNLP (§2), so its directions are judged on source similarity alone and must be reported that way.

| src_lang   | tgt_lang   |   rows |   median |    p05 |
|:-----------|:-----------|-------:|---------:|-------:|
| ben        | hin        |   1769 |   0.9592 | 0.8897 |
| ben        | tel        |   1769 |   0.9475 | 0.869  |
| hin        | ben        |   1769 |   0.9431 | 0.8654 |
| hin        | tel        |   1769 |   0.962  | 0.8943 |
| tel        | ben        |   1769 |   0.9341 | 0.8577 |
| tel        | hin        |   1769 |   0.9636 | 0.9013 |

## Rationale

- τ=0.82 rejects 0.0% of human translations of the same items — inside the 5% the calibration allows, so the threshold is not rejecting known-good work.
- No direction falls more than 20% below τ, the signal §8 names for a threshold measuring the wrong thing.
- This task's native sentences run 148 characters at the median. LaBSE cosine falls as context shortens, which is why τ is per task: the same threshold rejects 0% of human translations on task 2 and 23% on task 3, and the difference is text length, not translation quality.

226 marginal pairs either side of τ are exported to `tau_inspection_pairs.parquet` for reading — 30 per side per target language, chosen by closeness to τ, which are the pairs the threshold actually decides.
