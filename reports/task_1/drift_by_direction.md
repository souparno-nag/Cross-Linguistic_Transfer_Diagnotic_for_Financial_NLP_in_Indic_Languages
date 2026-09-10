# T-109 — translation drift by direction, task 1

τ = 0.82, config `d93963998a4be524`, 68358 MT pairs, every pair scored (T-108).

Below-τ rows carry `translation_drift` and **remain in the corpus** — §4 rule 1 makes a low-similarity pair a reported category, not garbage.

| src_lang   | tgt_lang   |   rows |   mean |   median |    p05 |    p25 |   below_tau |   below_share |
|:-----------|:-----------|-------:|-------:|---------:|-------:|-------:|------------:|--------------:|
| ben        | mal        |   6130 | 0.8336 |   0.8443 | 0.7202 | 0.8041 |        2090 |        0.3409 |
| ben        | tel        |   6130 | 0.8421 |   0.8488 | 0.743  | 0.8083 |        1930 |        0.3148 |
| ben        | hin        |   6130 | 0.8462 |   0.8541 | 0.7453 | 0.8101 |        1801 |        0.2938 |
| tel        | ben        |   6016 | 0.8738 |   0.9093 | 0.6306 | 0.8483 |        1188 |        0.1975 |
| tel        | mal        |   6016 | 0.8757 |   0.9058 | 0.661  | 0.8506 |        1133 |        0.1883 |
| tel        | hin        |   6016 | 0.8698 |   0.9163 | 0.5398 | 0.8662 |         934 |        0.1553 |
| hin        | mal        |  10640 | 0.8923 |   0.9062 | 0.7829 | 0.8677 |        1154 |        0.1085 |
| hin        | ben        |  10640 | 0.9173 |   0.9345 | 0.8036 | 0.8973 |         691 |        0.0649 |
| hin        | tel        |  10640 | 0.9225 |   0.9369 | 0.8243 | 0.9067 |         498 |        0.0468 |

Worst direction: `ben→mal` at 34.1% below τ (median <bound method Series.median of src_lang          ben
tgt_lang          mal
rows             6130
mean           0.8336
median         0.8443
p05            0.7202
p25            0.8041
below_tau        2090
below_share    0.3409
Name: 1, dtype: object>).

## By span recovered

| src_lang   | tgt_lang   | gold_class   |   rows |   median |   below_share |
|:-----------|:-----------|:-------------|-------:|---------:|--------------:|
| ben        | hin        | False        |   1160 |   0.8231 |        0.4698 |
| ben        | hin        | True         |   4970 |   0.8608 |        0.2527 |
| ben        | mal        | False        |   1451 |   0.8192 |        0.5024 |
| ben        | mal        | True         |   4679 |   0.8515 |        0.2909 |
| ben        | tel        | False        |   1061 |   0.8244 |        0.4637 |
| ben        | tel        | True         |   5069 |   0.8542 |        0.2837 |
| hin        | ben        | False        |   1630 |   0.8605 |        0.2804 |
| hin        | ben        | True         |   9010 |   0.941  |        0.026  |
| hin        | mal        | False        |   1841 |   0.8428 |        0.3764 |
| hin        | mal        | True         |   8799 |   0.9146 |        0.0524 |
| hin        | tel        | False        |   1289 |   0.8737 |        0.2498 |
| hin        | tel        | True         |   9351 |   0.9415 |        0.0188 |
| tel        | ben        | False        |   1141 |   0.7071 |        0.6889 |
| tel        | ben        | True         |   4875 |   0.9214 |        0.0825 |
| tel        | hin        | False        |   1012 |   0.7345 |        0.6136 |
| tel        | hin        | True         |   5004 |   0.9254 |        0.0625 |
| tel        | mal        | False        |   1149 |   0.7197 |        0.6527 |
| tel        | mal        | True         |   4867 |   0.9162 |        0.0787 |

## More than 20% below τ

T-110 treats this as evidence that the threshold is measuring the wrong thing rather than as a quality verdict on these directions:

- `ben→mal`: 34.1% below τ
- `ben→tel`: 31.5% below τ
- `ben→hin`: 29.4% below τ
