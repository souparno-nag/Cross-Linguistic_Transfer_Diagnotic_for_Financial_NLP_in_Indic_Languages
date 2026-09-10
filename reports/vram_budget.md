# VRAM budget — T-205

Device: NVIDIA GeForce RTX 3050 Laptop GPU, 3.7 GiB total.

Largest batch that survives forward → backward → optimiser step, `attention_mask` all ones (worst case).

| encoder | max_len | fp16 | max batch | peak GiB | note |
|---|---|---|---|---|---|
| indicbert-v2 | 64 | True | 64 | 1.60 |  |
| indicbert-v2 | 128 | True | 64 | 3.04 |  |
| indicbert-v2 | 256 | True | 32 | 3.04 |  |
| xlm-r-base | 64 | True | 64 | 2.76 |  |
| xlm-r-base | 128 | True | 32 | 2.76 |  |
| xlm-r-base | 256 | True | 12 | 2.37 |  |
| mbert-base | 64 | True | 64 | 2.38 |  |
| mbert-base | 128 | True | 48 | 3.32 |  |
| mbert-base | 256 | True | 24 | 3.16 |  |

## Full-epoch check

One epoch on the native training fold at each shipped config's settings.

| config | batch | max_len | grad_accum | fp16 | peak GiB | seconds | result |
|---|---|---|---|---|---|---|---|
| task2_hin_indicbert.yaml | 16 | 128 | 1 | True | 1.14 | 29 | ok |
| task3_hin_indicbert.yaml | 16 | 64 | 1 | True | 0.73 | 7 | ok |
