# T-105 — entity preservation, task 3

Source: corpus MT splits · 4788 rows · threshold 95%

| src_lang   | tgt_lang   |   rows |   preserved |   dropped |   altered |   currency_lost |   percent_lost |   corrupted | below_threshold   |
|:-----------|:-----------|-------:|------------:|----------:|----------:|----------------:|---------------:|------------:|:------------------|
| hin        | mal        |    531 |      0.774  |        65 |        52 |              78 |              7 |           1 | True              |
| ben        | mal        |    532 |      0.8102 |        79 |        60 |              56 |              4 |           0 | True              |
| tel        | mal        |    531 |      0.8606 |        61 |        50 |              19 |              5 |           1 | True              |
| hin        | tel        |    532 |      0.8835 |        20 |        13 |              45 |              6 |           0 | True              |
| tel        | ben        |    531 |      0.9153 |        34 |        24 |              16 |              4 |           1 | True              |
| tel        | hin        |    532 |      0.9192 |        34 |        22 |              14 |              7 |           0 | True              |
| ben        | tel        |    532 |      0.9267 |        26 |        12 |              24 |              4 |           0 | True              |
| hin        | ben        |    530 |      0.9358 |        14 |         8 |              21 |              6 |           2 | True              |
| ben        | hin        |    532 |      0.9699 |        23 |        10 |               1 |              5 |           0 | False             |

5 rows (0.1%) carry the §3.4 placeholder or escape corruption and are excluded from the rate above — their content was lost for reasons unrelated to numeral handling.

## Directions below threshold

- `hin→mal`: 77.4% preserved, 65 dropped, 52 altered
- `ben→mal`: 81.0% preserved, 79 dropped, 60 altered
- `tel→mal`: 86.1% preserved, 61 dropped, 50 altered
- `hin→tel`: 88.3% preserved, 20 dropped, 13 altered
- `tel→ben`: 91.5% preserved, 34 dropped, 24 altered
- `tel→hin`: 91.9% preserved, 34 dropped, 22 altered
- `ben→tel`: 92.7% preserved, 26 dropped, 12 altered
- `hin→ben`: 93.6% preserved, 14 dropped, 8 altered
