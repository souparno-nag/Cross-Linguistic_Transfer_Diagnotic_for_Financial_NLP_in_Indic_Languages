# T-104 — translation pipeline smoke test

Decoding fingerprint `08e5a3dc7f258d7e`, identical across all 9 directions so drift figures stay comparable between them.

```json
{
  "num_beams": 5,
  "max_length": 256,
  "min_length": 0,
  "length_penalty": 1.0,
  "do_sample": false,
  "num_return_sequences": 1,
  "use_cache": false
}
```

| model   | name                                 |   rows |   numerals_identical |   empty |   placeholder_leak |   escape_leak |   total_seconds |
|:--------|:-------------------------------------|-------:|---------------------:|--------:|-------------------:|--------------:|----------------:|
| 1B      | ai4bharat/indictrans2-indic-indic-1B |    180 |               0.8611 |       0 |                  0 |             5 |           105.8 |

20 sentences per direction. Per-sentence output is in `t104_smoke_*.parquet` for hand-checking.

`numerals_identical` counts rows whose digit set is unchanged. It is a rough signal, **not** an accuracy score: a correct translation can legitimately change the digits, because the two numbering systems differ. `১০০ মিলিয়ন` (100 million) → `10 करोड़` (10 crore) is correct and counts here as a mismatch. Scale-aware comparison is T-105.
