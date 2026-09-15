# Deferred training configs

Configs in this directory are **not** picked up by `scripts/t206_baseline.py`,
which globs `configs/train/*.yaml` non-recursively. They are kept because the
work that produced them is a result, not because they are waiting to run.

## `task{2,3}_hin_xlmr.yaml` — XLM-R, deferred at T-500 on VRAM grounds

XLM-R base cannot be fully fine-tuned on this project's GPU. Under AMP the
master weights stay fp32 and AdamW keeps two fp32 moments per parameter, so a
full fine-tune costs 16 bytes per parameter regardless of batch size or
sequence length:

| encoder | params | fixed cost | headroom on a 3.68 GiB card |
|---|---|---|---|
| `indicbert-v2` | 34M | 0.50 GiB | +3.18 GiB |
| `mbert-base` | 179M | 2.65 GiB | +1.03 GiB |
| `xlm-r-base` | 279M | **4.14 GiB** | **−0.46 GiB** |

That cost is incurred before a single activation, so it does not fit at batch
size 1 with a single token, and no reduction in `batch_size`, `max_len` or
gradient checkpointing can change it. Reproduce with:

    python -m scripts.t500_smoke --encoder-config configs/train/deferred/task2_hin_xlmr.yaml --audit-only

T-503's adapter fallback was considered and rejected: IndicBERT was fully
fine-tuned, so an adapter-trained XLM-R would confound the encoder with the
training method and make T-505's capacity-dilution question unanswerable.
Phase 5 therefore runs mBERT only, and the exclusion of XLM-R is reported as a
hardware limit with the arithmetic above rather than smoothed over
(CLAUDE5.md working agreement: "report a lost comparison honestly").

The files stay under test: `tests/test_t500_smoke.py` asserts they remain
protocol-identical to their IndicBERT siblings, so if the project ever moves to
a larger card they can be run without first re-deriving whether they drifted.
