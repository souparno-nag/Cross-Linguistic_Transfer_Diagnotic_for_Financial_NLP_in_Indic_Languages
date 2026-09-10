# T-110 — calibrating τ, task 1

τ in effect = 0.82. **Verdict: confirmed.**

## What MT scores

| src_lang   | tgt_lang   |   rows |   median |    p05 |   below_share |
|:-----------|:-----------|-------:|---------:|-------:|--------------:|
| ben        | mal        |   6130 |   0.8443 | 0.7202 |        0.3409 |
| ben        | tel        |   6130 |   0.8488 | 0.743  |        0.3148 |
| ben        | hin        |   6130 |   0.8541 | 0.7453 |        0.2938 |
| tel        | ben        |   6016 |   0.9093 | 0.6306 |        0.1975 |
| tel        | mal        |   6016 |   0.9058 | 0.661  |        0.1883 |
| tel        | hin        |   6016 |   0.9163 | 0.5398 |        0.1553 |
| hin        | mal        |  10640 |   0.9062 | 0.7829 |        0.1085 |
| hin        | ben        |  10640 |   0.9345 | 0.8036 |        0.0649 |
| hin        | tel        |  10640 |   0.9369 | 0.8243 |        0.0468 |

## Rationale

- Task 1 has no human reference in any direction — its native splits are independently sourced (§2.2), so there is no known-good pair to calibrate against. τ stays at the inherited value, and every figure derived from it must be read as inherited, not calibrated.
- The other tasks' calibrated values cannot be borrowed either: τ tracks text length as much as translation quality, and this task's text is a different length again.

240 marginal pairs either side of τ are exported to `tau_inspection_pairs.parquet` for reading — 30 per side per target language, chosen by closeness to τ, which are the pairs the threshold actually decides.
