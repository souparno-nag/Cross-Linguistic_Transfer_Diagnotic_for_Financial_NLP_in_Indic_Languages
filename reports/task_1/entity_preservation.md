# T-105 — entity preservation, task 1

Source: corpus MT splits · 68358 rows · threshold 95%

| src_lang   | tgt_lang   |   rows |   preserved |   dropped |   altered |   currency_lost |   percent_lost |   corrupted | below_threshold   |
|:-----------|:-----------|-------:|------------:|----------:|----------:|----------------:|---------------:|------------:|:------------------|
| ben        | mal        |   5698 |      0.4984 |      6159 |      2469 |             265 |            538 |         432 | True              |
| hin        | mal        |  10356 |      0.5688 |      8801 |      2353 |             849 |           2022 |         284 | True              |
| ben        | tel        |   5872 |      0.578  |      4861 |      3116 |             191 |            249 |         258 | True              |
| ben        | hin        |   5948 |      0.5891 |      5091 |      2613 |             138 |             34 |         182 | True              |
| hin        | tel        |  10442 |      0.6753 |      6564 |      2596 |             503 |            638 |         198 | True              |
| tel        | mal        |   5253 |      0.7076 |      2118 |      1282 |             319 |            460 |         763 | True              |
| hin        | ben        |  10057 |      0.7093 |      6036 |      2358 |             373 |            122 |         583 | True              |
| tel        | ben        |   5194 |      0.7668 |      1869 |       631 |             212 |            101 |         822 | True              |
| tel        | hin        |   5322 |      0.7867 |      1628 |       903 |             180 |             92 |         694 | True              |

4216 rows (6.2%) carry the §3.4 placeholder or escape corruption and are excluded from the rate above — their content was lost for reasons unrelated to numeral handling.

## Directions below threshold

- `ben→mal`: 49.8% preserved, 6159 dropped, 2469 altered
- `hin→mal`: 56.9% preserved, 8801 dropped, 2353 altered
- `ben→tel`: 57.8% preserved, 4861 dropped, 3116 altered
- `ben→hin`: 58.9% preserved, 5091 dropped, 2613 altered
- `hin→tel`: 67.5% preserved, 6564 dropped, 2596 altered
- `tel→mal`: 70.8% preserved, 2118 dropped, 1282 altered
- `hin→ben`: 70.9% preserved, 6036 dropped, 2358 altered
- `tel→ben`: 76.7% preserved, 1869 dropped, 631 altered
- `tel→hin`: 78.7% preserved, 1628 dropped, 903 altered
