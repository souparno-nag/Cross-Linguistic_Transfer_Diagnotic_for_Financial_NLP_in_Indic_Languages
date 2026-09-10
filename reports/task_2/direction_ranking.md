# T-111 — direction ranking, task 2

τ = 0.82. Ranked by the mean of each measure's rank, not by a weighted sum: a drift rate and a numeral-preservation rate are not on one scale, and inventing weights between them would manufacture precision. Rank averaging says only "worse on more measures".

|   rank | src_lang   | tgt_lang   | quadrant               |   median_sim |   drift_rate |   entity_preserved |   corrupt_rate |   truncation_rate |
|-------:|:-----------|:-----------|:-----------------------|-------------:|-------------:|-------------------:|---------------:|------------------:|
|      1 | ben        | hin        | Indo-Aryan->Indo-Aryan |       0.9309 |       0.0076 |             0.9403 |         0      |            0      |
|      2 | hin        | tel        | Indo-Aryan->Dravidian  |       0.9445 |       0.0022 |             0.916  |         0.0004 |            0      |
|      3 | tel        | hin        | Dravidian->Indo-Aryan  |       0.9432 |       0.0039 |             0.9078 |         0      |            0      |
|      4 | hin        | ben        | Indo-Aryan->Indo-Aryan |       0.9495 |       0.0031 |             0.9574 |         0.0031 |            0      |
|      5 | tel        | ben        | Dravidian->Indo-Aryan  |       0.9389 |       0.0068 |             0.9207 |         0.0024 |            0      |
|      6 | tel        | mal        | Dravidian->Dravidian   |       0.924  |       0.0159 |             0.9343 |         0.001  |            0      |
|      7 | ben        | tel        | Indo-Aryan->Dravidian  |       0.9229 |       0.0238 |             0.9197 |         0      |            0.0004 |
|      8 | hin        | mal        | Indo-Aryan->Dravidian  |       0.9075 |       0.0442 |             0.8953 |         0.0009 |            0      |
|      9 | ben        | mal        | Indo-Aryan->Dravidian  |       0.9016 |       0.0543 |             0.8976 |         0.0009 |            0.0004 |

## By typological quadrant

| quadrant               |   directions |   mean_drift |   mean_entity |
|:-----------------------|-------------:|-------------:|--------------:|
| Dravidian->Dravidian   |            1 |       0.0159 |        0.9343 |
| Dravidian->Indo-Aryan  |            2 |       0.0053 |        0.9142 |
| Indo-Aryan->Dravidian  |            4 |       0.0311 |        0.9071 |
| Indo-Aryan->Indo-Aryan |            2 |       0.0054 |        0.9489 |

## The questions

**Is Hi→Ml worse than Bn→Ml?** Better: Hi→Ml drifts on 4.4% of rows against Bn→Ml's 5.4% (ranks 8 and 9 of 9).

**Are Dravidian-source directions worse?** Mean drift 0.9% from Dravidian source against 2.3% from Indo-Aryan — no, the opposite.

**Is the Indo-Aryan↔Dravidian penalty symmetric?** No: crossing into Dravidian drifts on 3.1% of rows, crossing back into Indo-Aryan on 0.5% — a 2.6% gap in the same language pairs. The cost is in the direction of travel, not the pairing.

**Reading the drift column.** It counts pairs below τ, and T-110 found that τ=0.82 also rejects a quarter of *human* translations of the same items. So the column ranks directions against each other reliably; its absolute level is not a failure rate.
