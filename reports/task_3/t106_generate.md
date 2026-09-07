# T-106 — generated splits, task 3

Model `ai4bharat/indictrans2-indic-indic-1B`, decoding fingerprint `08e5a3dc7f258d7e`, identical across all directions so drift figures stay comparable (§8).

| block   | src_lang   | tgt_lang   |   rows |   unique_source_texts |   empty |   seconds |
|:--------|:-----------|:-----------|-------:|----------------------:|--------:|----------:|
| B       | ben        | hin        |    532 |                   532 |       0 |     211.2 |
| B       | ben        | tel        |    532 |                   532 |       0 |     174.3 |
| B       | ben        | mal        |    532 |                   532 |       0 |     257.9 |
| H       | hin        | ben        |    532 |                   532 |       0 |     185   |
| H       | hin        | tel        |    532 |                   532 |       0 |     264.3 |
| H       | hin        | mal        |    532 |                   532 |       0 |     269.3 |
| T       | tel        | hin        |    532 |                   532 |       0 |     276.3 |
| T       | tel        | ben        |    532 |                   532 |       0 |     181.3 |
| T       | tel        | mal        |    532 |                   532 |       0 |       0   |

`unique_source_texts` is what actually went to the model: identical sentences are translated once and the result shared, so one sentence carrying three annotated numbers cannot come back as three different sentences.
