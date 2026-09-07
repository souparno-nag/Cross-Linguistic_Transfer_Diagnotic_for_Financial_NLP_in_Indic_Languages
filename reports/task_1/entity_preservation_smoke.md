# T-105 — entity preservation, task 1

Source: T-104 smoke sample · 180 rows · threshold 95%

| src_lang   | tgt_lang   |   rows |   preserved |   dropped |   altered |   currency_lost |   percent_lost |   corrupted | below_threshold   |
|:-----------|:-----------|-------:|------------:|----------:|----------:|----------------:|---------------:|------------:|:------------------|
| ben        | tel        |     15 |      0.8667 |         2 |         0 |               0 |              0 |           5 | True              |
| hin        | tel        |     20 |      0.9    |         1 |         0 |               1 |              0 |           0 | True              |
| tel        | mal        |     20 |      0.9    |         0 |         0 |               0 |              2 |           0 | True              |
| ben        | mal        |     15 |      0.9333 |         0 |         0 |               0 |              1 |           5 | True              |
| hin        | ben        |     18 |      0.9444 |         1 |         0 |               0 |              0 |           2 | True              |
| tel        | hin        |     20 |      0.95   |         1 |         1 |               0 |              0 |           0 | False             |
| hin        | mal        |     20 |      0.95   |         1 |         0 |               0 |              0 |           0 | False             |
| ben        | hin        |     14 |      1      |         0 |         0 |               0 |              0 |           6 | False             |
| tel        | ben        |     20 |      1      |         0 |         0 |               0 |              0 |           0 | False             |

18 rows (10.0%) carry the §3.4 placeholder or escape corruption and are excluded from the rate above — their content was lost for reasons unrelated to numeral handling.

## Directions below threshold

- `ben→tel`: 86.7% preserved, 2 dropped, 0 altered
- `hin→tel`: 90.0% preserved, 1 dropped, 0 altered
- `tel→mal`: 90.0% preserved, 0 dropped, 0 altered
- `ben→mal`: 93.3% preserved, 0 dropped, 0 altered
- `hin→ben`: 94.4% preserved, 1 dropped, 0 altered
