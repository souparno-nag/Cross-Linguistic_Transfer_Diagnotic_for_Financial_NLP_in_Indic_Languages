# T-106 — generated splits, task 1

Model `ai4bharat/indictrans2-indic-indic-1B`, decoding fingerprint `08e5a3dc7f258d7e`, identical across all directions so drift figures stay comparable (§8).

| block   | src_lang   | tgt_lang   |   rows |   unique_source_texts |   empty |   span_recovered |   span_recovery_rate |   span_scale_shift |   span_ambiguous |   seconds |
|:--------|:-----------|:-----------|-------:|----------------------:|--------:|-----------------:|---------------------:|-------------------:|-----------------:|----------:|
| B       | ben        | hin        |   6130 |                  3412 |       0 |             4970 |               0.8108 |                 21 |              206 |         0 |
| B       | ben        | tel        |   6130 |                  3412 |       0 |             5069 |               0.8269 |                 24 |              193 |         0 |
| B       | ben        | mal        |   6130 |                  3412 |       0 |             4679 |               0.7633 |                 34 |              187 |         0 |
| H       | hin        | ben        |  10640 |                  6357 |       0 |             9010 |               0.8468 |                 32 |              460 |         0 |
| H       | hin        | tel        |  10640 |                  6357 |       0 |             9351 |               0.8789 |                 48 |              465 |         0 |
| H       | hin        | mal        |  10640 |                  6357 |       0 |             8799 |               0.827  |                 59 |              435 |         0 |
| T       | tel        | hin        |   6016 |                  3698 |       0 |             5004 |               0.8318 |                 37 |              208 |         0 |
| T       | tel        | ben        |   6016 |                  3698 |       0 |             4875 |               0.8103 |                 14 |              218 |         0 |
| T       | tel        | mal        |   6016 |                  3698 |       0 |             4867 |               0.809  |                 30 |              194 |         0 |

`unique_source_texts` is what actually went to the model: identical sentences are translated once and the result shared, so one sentence carrying three annotated numbers cannot come back as three different sentences.

`span_recovered` counts MT rows where the annotated number was found again in the translation. A failure is flagged and kept (§4 rule 1); the rate is a headline result, not an error. `span_scale_shift` separates numbers rewritten as an equivalent quantity (`১০০ মিলিয়ন` → `10 करोड़`) from numbers genuinely lost.
