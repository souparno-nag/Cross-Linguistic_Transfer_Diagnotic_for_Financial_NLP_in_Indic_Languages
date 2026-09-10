# T-110 — calibrating τ, task 3

Inherited τ = 0.82. **Verdict: revised, revised to 0.73.**

## What MT scores

| src_lang   | tgt_lang   |   rows |   median |    p05 |   below_share |
|:-----------|:-----------|-------:|---------:|-------:|--------------:|
| hin        | mal        |    532 |   0.8588 | 0.7372 |        0.297  |
| tel        | mal        |    532 |   0.8607 | 0.7458 |        0.2613 |
| ben        | mal        |    532 |   0.8702 | 0.762  |        0.2124 |
| tel        | ben        |    532 |   0.8812 | 0.7745 |        0.1617 |
| ben        | tel        |    532 |   0.8889 | 0.7769 |        0.1241 |
| tel        | hin        |    532 |   0.8905 | 0.7855 |        0.1015 |
| hin        | tel        |    532 |   0.9028 | 0.8008 |        0.0846 |
| ben        | hin        |    532 |   0.8984 | 0.7982 |        0.0771 |
| hin        | ben        |    532 |   0.9083 | 0.8107 |        0.0714 |

## What a human translation scores, on the same measure

The native splits of this task are human translations of one another (§2.1), so the cosine between two of them for the same item is a known-good pair measured exactly like an MT pair. This is the ceiling.

| pair    |   rows |   median |    p05 |   below_tau |
|:--------|-------:|---------:|-------:|------------:|
| ben-hin |    532 |   0.8758 | 0.7484 |      0.2124 |
| ben-tel |    532 |   0.8511 | 0.7084 |      0.3252 |
| hin-tel |    532 |   0.8839 | 0.7492 |      0.1654 |

## MT against a human translation of the same item

Same language, same item, so this isolates translation quality from cross-language drift. No Malayalam row appears: there is no native Malayalam anywhere in IndicFinNLP (§2), so its directions are judged on source similarity alone and must be reported that way.

| src_lang   | tgt_lang   |   rows |   median |    p05 |
|:-----------|:-----------|-------:|---------:|-------:|
| ben        | hin        |    532 |   0.9296 | 0.7992 |
| ben        | tel        |    532 |   0.8891 | 0.7346 |
| hin        | ben        |    532 |   0.9143 | 0.7599 |
| hin        | tel        |    532 |   0.9078 | 0.7624 |
| tel        | ben        |    532 |   0.9024 | 0.7462 |
| tel        | hin        |    532 |   0.9364 | 0.8205 |

## Rationale

- τ=0.82 rejects 23.4% of human translations of the same items, scored on exactly the same measure. A threshold that calls one human translation in 4 a failure is not measuring translation quality.
- 3 of 9 directions fall more than 20% below τ (hin→mal, tel→mal, ben→mal), which §8 names as the signal that the threshold is wrong rather than the translations.
- Revised τ = 0.73, the value that passes 95% of human translations. It is derived from the corpus rather than inherited, and it is a floor for 'plausibly a translation of this', not a quality score.

240 marginal pairs either side of τ are exported to `tau_inspection_pairs.parquet` for reading — 30 per side per target language, chosen by closeness to τ, which are the pairs the threshold actually decides.
