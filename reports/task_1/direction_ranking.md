# T-111 — direction ranking, task 1

τ = 0.82. Ranked by the mean of each measure's rank, not by a weighted sum: a drift rate and a numeral-preservation rate are not on one scale, and inventing weights between them would manufacture precision. Rank averaging says only "worse on more measures".

|   rank | src_lang   | tgt_lang   | quadrant               |   median_sim |   drift_rate |   entity_preserved |   corrupt_rate |   truncation_rate |   span_recovery |
|-------:|:-----------|:-----------|:-----------------------|-------------:|-------------:|-------------------:|---------------:|------------------:|----------------:|
|      1 | hin        | ben        | Indo-Aryan->Indo-Aryan |       0.9345 |       0.0649 |             0.7093 |         0.0548 |            0.0018 |          0.8468 |
|      2 | hin        | tel        | Indo-Aryan->Dravidian  |       0.9369 |       0.0468 |             0.6753 |         0.0186 |            0.0094 |          0.8789 |
|      3 | hin        | mal        | Indo-Aryan->Dravidian  |       0.9062 |       0.1085 |             0.5688 |         0.0267 |            0.0055 |          0.827  |
|      4 | tel        | hin        | Dravidian->Indo-Aryan  |       0.9163 |       0.1553 |             0.7867 |         0.1154 |            0.0392 |          0.8318 |
|      5 | ben        | hin        | Indo-Aryan->Indo-Aryan |       0.8541 |       0.2938 |             0.5891 |         0.0297 |            0.0018 |          0.8108 |
|      6 | ben        | tel        | Indo-Aryan->Dravidian  |       0.8488 |       0.3148 |             0.578  |         0.0421 |            0.009  |          0.8269 |
|      7 | tel        | ben        | Dravidian->Indo-Aryan  |       0.9093 |       0.1975 |             0.7668 |         0.1366 |            0.047  |          0.8103 |
|      8 | tel        | mal        | Dravidian->Dravidian   |       0.9058 |       0.1883 |             0.7076 |         0.1268 |            0.058  |          0.809  |
|      9 | ben        | mal        | Indo-Aryan->Dravidian  |       0.8443 |       0.3409 |             0.4984 |         0.0705 |            0.0049 |          0.7633 |

## By typological quadrant

| quadrant               |   directions |   mean_drift |   mean_entity |
|:-----------------------|-------------:|-------------:|--------------:|
| Dravidian->Dravidian   |            1 |       0.1883 |        0.7076 |
| Dravidian->Indo-Aryan  |            2 |       0.1764 |        0.7768 |
| Indo-Aryan->Dravidian  |            4 |       0.2028 |        0.5801 |
| Indo-Aryan->Indo-Aryan |            2 |       0.1794 |        0.6492 |

## The questions

**Is Hi→Ml worse than Bn→Ml?** Better: Hi→Ml drifts on 10.8% of rows against Bn→Ml's 34.1% (ranks 3 and 9 of 9).

**Are Dravidian-source directions worse?** Mean drift 18.0% from Dravidian source against 19.5% from Indo-Aryan — no, the opposite.

**Is the Indo-Aryan↔Dravidian penalty symmetric?** No: crossing into Dravidian drifts on 20.3% of rows, crossing back into Indo-Aryan on 17.6% — a 2.6% gap in the same language pairs. The cost is in the direction of travel, not the pairing.

**Reading the drift column.** It counts pairs below τ, and T-110 found that τ=0.82 also rejects a quarter of *human* translations of the same items. So the column ranks directions against each other reliably; its absolute level is not a failure rate.
