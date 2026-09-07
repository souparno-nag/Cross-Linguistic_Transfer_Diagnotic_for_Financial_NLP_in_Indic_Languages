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

| model   | name                                        |   rows |   numerals_preserved |   empty |   total_seconds |
|:--------|:--------------------------------------------|-------:|---------------------:|--------:|----------------:|
| 1B      | ai4bharat/indictrans2-indic-indic-1B        |    180 |               0.9944 |       0 |            61.2 |
| 320M    | ai4bharat/indictrans2-indic-indic-dist-320M |    180 |               1      |       0 |            18.4 |

20 sentences per direction. Per-sentence output is in `t104_smoke_*.parquet` for hand-checking — numeral preservation is a cheap proxy, not a substitute for reading the translations.
