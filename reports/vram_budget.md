# VRAM budget — T-205

Device: NVIDIA GeForce RTX 3050 Laptop GPU, 3.7 GiB total.

Largest batch that survives forward → backward → optimiser step, `attention_mask` all ones (worst case).

**floor** is the fixed cost of fp32 weights + gradients + AdamW's two moments — 16 bytes per parameter, independent of batch size and sequence length. **activations** is what the measured peak leaves over it, and is the only part a batch sweep can move. A peak *below* the floor is impossible for a completed optimiser step and means the measurement is wrong (see `src/vram.py`, the T-500 note).

| encoder | max_len | fp16 | max batch | peak GiB | floor GiB | activations GiB | note |
|---|---|---|---|---|---|---|---|
| indicbert-v2 | 64 | True | 64 | 1.60 | 0.50 | 1.10 |  |
| indicbert-v2 | 128 | True | 64 | 3.04 | 0.50 | 2.54 |  |
| indicbert-v2 | 256 | True | 32 | 3.04 | 0.50 | 2.54 |  |
| xlm-r-base | 64 | True | 0 | — | 4.14 | — | does not fit at any batch size: weights + gradients + AdamW state need 4.14 GiB before any activation, and the device has 3.68 GiB |
| xlm-r-base | 128 | True | 0 | — | 4.14 | — | does not fit at any batch size: weights + gradients + AdamW state need 4.14 GiB before any activation, and the device has 3.68 GiB |
| xlm-r-base | 256 | True | 0 | — | 4.14 | — | does not fit at any batch size: weights + gradients + AdamW state need 4.14 GiB before any activation, and the device has 3.68 GiB |
| mbert-base | 64 | True | 64 | 3.32 | 2.65 | 0.67 |  |
| mbert-base | 128 | True | 48 | 3.32 | 2.65 | 0.67 |  |
| mbert-base | 256 | True | 24 | 3.32 | 2.65 | 0.67 |  |

## Full-epoch check

One epoch on the native training fold at each shipped config's settings.

| config | batch | max_len | grad_accum | fp16 | peak GiB | seconds | result |
|---|---|---|---|---|---|---|---|
| task2_hin_indicbert.yaml | 16 | 192 | 1 | True | 1.50 | 31 | ok |
| task2_hin_mbert.yaml | 16 | 192 | 1 | True | 3.33 | 22 | OOM: CUDA out of memory. Tried to allocate 352.00 MiB. GPU 0 has a total capacity of 3.68 GiB of which 354.00 MiB is free. |
| task3_hin_indicbert.yaml | 16 | 64 | 1 | True | 0.73 | 7 | ok |
| task3_hin_mbert.yaml | 16 | 64 | 1 | True | 3.32 | 8 | ok |
