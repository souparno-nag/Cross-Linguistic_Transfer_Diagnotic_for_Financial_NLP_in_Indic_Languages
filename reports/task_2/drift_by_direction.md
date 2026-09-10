# T-109 — translation drift by direction, task 2

τ = 0.82, config `d93963998a4be524`, 19614 MT pairs, every pair scored (T-108).

Below-τ rows carry `translation_drift` and **remain in the corpus** — §4 rule 1 makes a low-similarity pair a reported category, not garbage.

| src_lang   | tgt_lang   |   rows |   mean |   median |    p05 |    p25 |   below_tau |   below_share |
|:-----------|:-----------|-------:|-------:|---------:|-------:|-------:|------------:|--------------:|
| ben        | mal        |   2228 | 0.8959 |   0.9016 | 0.8174 | 0.8721 |         121 |        0.0543 |
| hin        | mal        |   2238 | 0.9015 |   0.9075 | 0.8237 | 0.8801 |          99 |        0.0442 |
| ben        | tel        |   2228 | 0.9157 |   0.9229 | 0.8447 | 0.8925 |          53 |        0.0238 |
| tel        | mal        |   2072 | 0.9175 |   0.924  | 0.8476 | 0.8993 |          33 |        0.0159 |
| ben        | hin        |   2228 | 0.9252 |   0.9309 | 0.8599 | 0.9058 |          17 |        0.0076 |
| tel        | ben        |   2072 | 0.9327 |   0.9389 | 0.8758 | 0.9173 |          14 |        0.0068 |
| tel        | hin        |   2072 | 0.9372 |   0.9432 | 0.8847 | 0.9235 |           8 |        0.0039 |
| hin        | ben        |   2238 | 0.9438 |   0.9495 | 0.895  | 0.9322 |           7 |        0.0031 |
| hin        | tel        |   2238 | 0.9386 |   0.9445 | 0.8813 | 0.9263 |           5 |        0.0022 |

Worst direction: `ben→mal` at 5.4% below τ (median <bound method Series.median of src_lang          ben
tgt_lang          mal
rows             2228
mean           0.8959
median         0.9016
p05            0.8174
p25            0.8721
below_tau         121
below_share    0.0543
Name: 1, dtype: object>).

## By gold class

| src_lang   | tgt_lang   | gold_class    |   rows |   median |   below_share |
|:-----------|:-----------|:--------------|-------:|---------:|--------------:|
| ben        | hin        | sustainable   |   1203 |   0.932  |        0.005  |
| ben        | hin        | unsustainable |   1025 |   0.9294 |        0.0107 |
| ben        | mal        | sustainable   |   1203 |   0.9053 |        0.0449 |
| ben        | mal        | unsustainable |   1025 |   0.8981 |        0.0654 |
| ben        | tel        | sustainable   |   1203 |   0.9257 |        0.0133 |
| ben        | tel        | unsustainable |   1025 |   0.9174 |        0.0361 |
| hin        | ben        | sustainable   |   1212 |   0.952  |        0.0008 |
| hin        | ben        | unsustainable |   1026 |   0.9463 |        0.0058 |
| hin        | mal        | sustainable   |   1212 |   0.9133 |        0.0305 |
| hin        | mal        | unsustainable |   1026 |   0.9    |        0.0604 |
| hin        | tel        | sustainable   |   1212 |   0.9484 |        0.0008 |
| hin        | tel        | unsustainable |   1026 |   0.9383 |        0.0039 |
| tel        | ben        | sustainable   |   1119 |   0.9431 |        0.0009 |
| tel        | ben        | unsustainable |    953 |   0.9336 |        0.0136 |
| tel        | hin        | sustainable   |   1119 |   0.9475 |        0      |
| tel        | hin        | unsustainable |    953 |   0.9374 |        0.0084 |
| tel        | mal        | sustainable   |   1119 |   0.9272 |        0.0143 |
| tel        | mal        | unsustainable |    953 |   0.9195 |        0.0178 |
