# T-503 — MAD-X adapter fallback: closed as unnecessary

**Status: closed without an adapter run.** CLAUDE5.md's own criterion is "adapter
run completes, **or** the task is closed as unnecessary with a note." This is the
note.

## The contingency did fire

T-503 is conditional on T-501 or T-502 being VRAM-blocked, and both were, in
different ways and with different outcomes:

| task | what was blocked | resolved by |
|---|---|---|
| T-501, XLM-R | every configuration, at every batch size | **not resolved** — deferred |
| T-502, mBERT | task 2 only, at `batch_size 16 / max_len 192` | micro-batching, no adapters needed |

**T-502's block was not the kind adapters fix.** mBERT ran out of room for
*activations*, so re-shaping the batch cleared it: `batch_size 8` with
`grad_accum 2` keeps the effective batch at 16 and therefore the same gradient,
since the loss is a mean. Measured at 3.41 GiB peak over a full epoch, in a
clean process. Reaching for adapters there would have introduced a protocol
difference to solve a problem that a supported knob already solved.

**T-501's block is different in kind and adapters would in fact fix it.** XLM-R
needs 4.14 GiB of fp32 weights, gradients and AdamW state before a single
activation, on a 3.68 GiB card — a fixed parameter cost, unmovable by batch
size. Freezing the base model removes the gradients and optimiser state for
278M parameters and leaves roughly 1.1 GiB plus small adapter matrices, which
fits comfortably. So this note is **not** claiming adapters were infeasible.
They were feasible and were rejected anyway.

## Why they were rejected

**1. They would have destroyed the comparison they were meant to rescue.**
IndicBERT-v2 and mBERT were fully fine-tuned. An adapter-trained XLM-R would
differ from both in *how it was trained* as well as in what it was pretrained
on, so a lower XLM-R score could mean capacity dilution or could mean adapters
have less room to adapt — and nothing in the results would distinguish them.
T-505's question is precisely whether multilinguality or capacity explains the
ranking; adding a third explanation makes it strictly less answerable. CLAUDE5.md
hard rule 1 says a comparison is only valid if nothing else moved.

**2. Removing that confound costs more than the arm is worth.** The fix would be
to run all three encoders with adapters, which means retraining IndicBERT-v2's
Phase 2 baselines, re-running its Phase 3 conditions, and re-validating the
T-207 gate that Phase 3 depends on — reopening two completed, committed phases
for an extension CLAUDE5.md itself calls deferrable, on a GPU where Phase 6 has
priority.

**3. The dependency risk is disproportionate.** MAD-X as specified (Pfeiffer et
al. [17], language *and* task adapters) means AdapterHub's `adapters` package,
which pins itself against particular `transformers` versions. This project's
`transformers==4.57.6` is load-bearing: it is pinned because IndicTrans2's
released checkpoints carry remote modelling code that breaks on 5.x in three
places, one of them unshimmable (CLAUDE.md §3.2), and that is the code the
entire frozen corpus was built with. Risking the environment that produced
`data/v1.0/` for a deferrable comparison arm is a bad trade.

## What was done instead, and what it cost

XLM-R is deferred rather than adapted. Its configs are kept, protocol-identical
to their IndicBERT siblings and still under test, in `configs/train/deferred/`
with the arithmetic recorded. Phase 5 ran two encoders on an identical protocol
rather than three on a mixed one.

The cost is real and is stated in T-505 rather than smoothed over: mBERT and
XLM-R have **identical 85.6M-parameter transformer bodies** and differ almost
entirely in vocabulary size and pretraining breadth, which makes that pair the
capacity-held-constant comparison capacity dilution actually calls for. Losing
XLM-R is therefore not losing a third data point — it is losing the controlled
experiment. T-505 records its question as left open for that reason.

## If this is revisited

On a card with 8 GB or more, `python -m scripts.t206_baseline --config
configs/train/deferred/task2_hin_xlmr.yaml` runs the deferred arm on the
original protocol, with no adapters and no confound. That is the version worth
waiting for. Adapters remain the fallback only if the hardware constraint is
permanent **and** all three encoders are re-run under them, with the protocol
difference stated in every table that reports the result.
