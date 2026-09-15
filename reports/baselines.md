# In-language baselines — T-206

indicbert-v2 fine-tuned on each native split, three seeds. Numbers are on the held-out **test** fold.

T-207's gate compares only the configs named in `configs/published_baselines.json`. IndicFinNLP publishes an IndicBERT number and nothing to compare another encoder against, so a Phase 5 encoder appears in this table without being gated against a published value. Comparing encoders to each other is T-504's job, not this report's.

**Effective batch is shown because it has to match for a cross-encoder comparison to mean anything** (CLAUDE5.md hard rule 1). Where two rows share an effective batch but differ in `batch × accum`, one was re-shaped because it did not fit in VRAM: that changes how many rows sit on the card at once, not the gradient, since the loss is a mean.

| config | encoder | split | batch × accum | effective batch | seeds | macro-F1 (mean ± std) | accuracy (mean ± std) |
|---|---|---|---|---|---|---|---|
| task2_hin_indicbert.yaml | indicbert-v2 | task2/H/hin/native | 16 × 1 | 16 | 0,1,2 | 0.8226 ± 0.0044 | 0.8244 ± 0.0052 |
| task3_hin_indicbert.yaml | indicbert-v2 | task3/H/hin/native | 16 × 1 | 16 | 0,1,2 | 0.1526 ± 0.0249 | 0.2250 ± 0.0331 |

## Per-seed macro-F1

| config | encoder | seed | macro-F1 | accuracy |
|---|---|---|---|---|
| task2_hin_indicbert.yaml | indicbert-v2 | 0 | 0.8194 | 0.8214 |
| task2_hin_indicbert.yaml | indicbert-v2 | 1 | 0.8209 | 0.8214 |
| task2_hin_indicbert.yaml | indicbert-v2 | 2 | 0.8277 | 0.8304 |
| task3_hin_indicbert.yaml | indicbert-v2 | 0 | 0.1620 | 0.2375 |
| task3_hin_indicbert.yaml | indicbert-v2 | 1 | 0.1714 | 0.2500 |
| task3_hin_indicbert.yaml | indicbert-v2 | 2 | 0.1243 | 0.1875 |
