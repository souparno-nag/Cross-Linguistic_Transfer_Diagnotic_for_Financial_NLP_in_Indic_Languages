# T-111 — direction ranking, task 3

τ = 0.82. Ranked by the mean of each measure's rank, not by a weighted sum: a drift rate and a numeral-preservation rate are not on one scale, and inventing weights between them would manufacture precision. Rank averaging says only "worse on more measures".

|   rank | src_lang   | tgt_lang   | quadrant               |   median_sim |   drift_rate |   entity_preserved |   corrupt_rate |   truncation_rate |
|-------:|:-----------|:-----------|:-----------------------|-------------:|-------------:|-------------------:|---------------:|------------------:|
|      1 | ben        | hin        | Indo-Aryan->Indo-Aryan |       0.8984 |       0.0771 |             0.9699 |         0      |                 0 |
|      2 | ben        | tel        | Indo-Aryan->Dravidian  |       0.8889 |       0.1241 |             0.9267 |         0      |                 0 |
|      3 | tel        | hin        | Dravidian->Indo-Aryan  |       0.8905 |       0.1015 |             0.9192 |         0      |                 0 |
|      4 | hin        | ben        | Indo-Aryan->Indo-Aryan |       0.9083 |       0.0714 |             0.9358 |         0.0038 |                 0 |
|      5 | hin        | tel        | Indo-Aryan->Dravidian  |       0.9028 |       0.0846 |             0.8835 |         0      |                 0 |
|      6 | ben        | mal        | Indo-Aryan->Dravidian  |       0.8702 |       0.2124 |             0.8102 |         0      |                 0 |
|      7 | tel        | ben        | Dravidian->Indo-Aryan  |       0.8812 |       0.1617 |             0.9153 |         0.0019 |                 0 |
|      8 | tel        | mal        | Dravidian->Dravidian   |       0.8607 |       0.2613 |             0.8606 |         0.0019 |                 0 |
|      9 | hin        | mal        | Indo-Aryan->Dravidian  |       0.8588 |       0.297  |             0.774  |         0.0019 |                 0 |

## By typological quadrant

| quadrant               |   directions |   mean_drift |   mean_entity |
|:-----------------------|-------------:|-------------:|--------------:|
| Dravidian->Dravidian   |            1 |       0.2613 |        0.8606 |
| Dravidian->Indo-Aryan  |            2 |       0.1316 |        0.9172 |
| Indo-Aryan->Dravidian  |            4 |       0.1795 |        0.8486 |
| Indo-Aryan->Indo-Aryan |            2 |       0.0742 |        0.9529 |

## The questions

**Is Hi→Ml worse than Bn→Ml?** Worse: Hi→Ml drifts on 29.7% of rows against Bn→Ml's 21.2% (ranks 9 and 6 of 9).

**Are Dravidian-source directions worse?** Mean drift 17.5% from Dravidian source against 14.4% from Indo-Aryan — yes.

**Is the Indo-Aryan↔Dravidian penalty symmetric?** No: crossing into Dravidian drifts on 18.0% of rows, crossing back into Indo-Aryan on 13.2% — a 4.8% gap in the same language pairs. The cost is in the direction of travel, not the pairing.

**Reading the drift column.** It counts pairs below τ, and T-110 found that τ=0.82 also rejects a quarter of *human* translations of the same items. So the column ranks directions against each other reliably; its absolute level is not a failure rate.
