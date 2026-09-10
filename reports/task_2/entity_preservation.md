# T-105 — entity preservation, task 2

Source: corpus MT splits · 19614 rows · threshold 95%

| src_lang   | tgt_lang   |   rows |   preserved |   dropped |   altered |   currency_lost |   percent_lost |   corrupted | below_threshold   |
|:-----------|:-----------|-------:|------------:|----------:|----------:|----------------:|---------------:|------------:|:------------------|
| hin        | mal        |   2236 |      0.8953 |       130 |        28 |             103 |             80 |           2 | True              |
| ben        | mal        |   2226 |      0.8976 |       340 |       127 |              53 |             74 |           2 | True              |
| tel        | hin        |   2072 |      0.9078 |       121 |        24 |              98 |             14 |           0 | True              |
| hin        | tel        |   2237 |      0.916  |        90 |        32 |              92 |             34 |           1 | True              |
| ben        | tel        |   2228 |      0.9197 |       289 |       127 |              42 |             39 |           0 | True              |
| tel        | ben        |   2067 |      0.9207 |        76 |        23 |             108 |              9 |           5 | True              |
| tel        | mal        |   2070 |      0.9343 |       120 |        25 |              29 |             35 |           2 | True              |
| ben        | hin        |   2228 |      0.9403 |       313 |       159 |               0 |              3 |           0 | True              |
| hin        | ben        |   2231 |      0.9574 |        64 |        29 |              51 |              5 |           7 | False             |

19 rows (0.1%) carry the §3.4 placeholder or escape corruption and are excluded from the rate above — their content was lost for reasons unrelated to numeral handling.

## Directions below threshold

- `hin→mal`: 89.5% preserved, 130 dropped, 28 altered
- `ben→mal`: 89.8% preserved, 340 dropped, 127 altered
- `tel→hin`: 90.8% preserved, 121 dropped, 24 altered
- `hin→tel`: 91.6% preserved, 90 dropped, 32 altered
- `ben→tel`: 92.0% preserved, 289 dropped, 127 altered
- `tel→ben`: 92.1% preserved, 76 dropped, 23 altered
- `tel→mal`: 93.4% preserved, 120 dropped, 25 altered
- `ben→hin`: 94.0% preserved, 313 dropped, 159 altered
