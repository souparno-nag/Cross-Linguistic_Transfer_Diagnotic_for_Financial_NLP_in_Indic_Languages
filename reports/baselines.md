# In-language baselines — T-206

indicbert-v2 and mbert-base fine-tuned on each native split, three seeds. Numbers are on the held-out **test** fold.

T-207's gate compares only the configs named in `configs/published_baselines.json`. IndicFinNLP publishes an IndicBERT number and nothing to compare another encoder against, so a Phase 5 encoder appears in this table without being gated against a published value. Comparing encoders to each other is T-504's job, not this report's.

**Effective batch is shown because it has to match for a cross-encoder comparison to mean anything** (CLAUDE5.md hard rule 1). Where two rows share an effective batch but differ in `batch × accum`, one was re-shaped because it did not fit in VRAM: that changes how many rows sit on the card at once, not the gradient, since the loss is a mean.

**Where a config has a run that never fit its training data, the converged-seed figure is shown alongside** — never instead of. The all-seeds mean stays the headline (`hard rule 3`), because a run that failed to train is still a run that happened and the instability is part of the result. `src/convergence.py` sets and justifies the threshold (peak **train** macro-F1 ≥ 0.5); it is deliberately far below what a healthy run reaches, and it is read from the run's own checkpoint.

| config | encoder | split | batch × accum | effective batch | seeds | macro-F1 (mean ± std) | converged seeds only | accuracy (mean ± std) |
|---|---|---|---|---|---|---|---|---|
| task2_ben_indicbert.yaml | indicbert-v2 | task2/B/ben/native | 16 × 1 | 16 | 0,1,2 | 0.8201 ± 0.0279 | all seeds converged | 0.8206 ± 0.0280 |
| task2_hin_indicbert.yaml | indicbert-v2 | task2/H/hin/native | 16 × 1 | 16 | 0,1,2 | 0.8226 ± 0.0044 | all seeds converged | 0.8244 ± 0.0052 |
| task2_hin_mbert.yaml | mbert-base | task2/H/hin/native | 8 × 2 | 16 | 0,1,2 | 0.8742 ± 0.0170 | all seeds converged | 0.8765 ± 0.0169 |
| task3_ben_indicbert.yaml | indicbert-v2 | task3/B/ben/native | 16 × 1 | 16 | 0,1,2 | 0.1269 ± 0.0549 | **0.1585 ± 0.0013** (n=2) | 0.2167 ± 0.0072 |
| task3_hin_indicbert.yaml | indicbert-v2 | task3/H/hin/native | 16 × 1 | 16 | 0,1,2 | 0.1526 ± 0.0249 | all seeds converged | 0.2250 ± 0.0331 |
| task3_hin_mbert.yaml | mbert-base | task3/H/hin/native | 16 × 1 | 16 | 0,1,2 | 0.3354 ± 0.0417 | all seeds converged | 0.4167 ± 0.0361 |

## Runs that did not fit their training data

These completed and logged a number without the optimisation ever getting going. Their **test** score is indistinguishable from a healthy run's on a task at its data ceiling, which is exactly why the training fold is what gets checked. They are kept in the corpus of results and in the all-seeds mean above; this section exists so the mean can be read knowing they are in it.

| run | peak train macro-F1 | test macro-F1 | epochs | verdict |
|---|---|---|---|---|
| `task3_ben_indicbert_seed0` | 0.1519 | 0.0635 | 13 | below 0.5 — did not train |

## Per-seed macro-F1

`peak train` is the run's best macro-F1 on its **own training fold**; a blank means the checkpoint is no longer on disk, which is unknown rather than failed.

| config | encoder | seed | macro-F1 | accuracy | peak train | trained? |
|---|---|---|---|---|---|---|
| task2_ben_indicbert.yaml | indicbert-v2 | 0 | 0.8292 | 0.8296 | 0.9972 | yes |
| task2_ben_indicbert.yaml | indicbert-v2 | 1 | 0.8422 | 0.8430 | 0.9966 | yes |
| task2_ben_indicbert.yaml | indicbert-v2 | 2 | 0.7888 | 0.7892 | 0.9932 | yes |
| task2_hin_indicbert.yaml | indicbert-v2 | 0 | 0.8194 | 0.8214 | 0.9989 | yes |
| task2_hin_indicbert.yaml | indicbert-v2 | 1 | 0.8209 | 0.8214 | 0.9955 | yes |
| task2_hin_indicbert.yaml | indicbert-v2 | 2 | 0.8277 | 0.8304 | 0.9871 | yes |
| task2_hin_mbert.yaml | mbert-base | 0 | 0.8809 | 0.8839 | 1.0000 | yes |
| task2_hin_mbert.yaml | mbert-base | 1 | 0.8549 | 0.8571 | 0.9994 | yes |
| task2_hin_mbert.yaml | mbert-base | 2 | 0.8870 | 0.8884 | 1.0000 | yes |
| task3_ben_indicbert.yaml | indicbert-v2 | 0 | 0.0635 | 0.2125 | 0.1519 | **no** |
| task3_ben_indicbert.yaml | indicbert-v2 | 1 | 0.1576 | 0.2125 | 0.9968 | yes |
| task3_ben_indicbert.yaml | indicbert-v2 | 2 | 0.1595 | 0.2250 | 0.6882 | yes |
| task3_hin_indicbert.yaml | indicbert-v2 | 0 | 0.1620 | 0.2375 | 0.9955 | yes |
| task3_hin_indicbert.yaml | indicbert-v2 | 1 | 0.1714 | 0.2500 | 0.9705 | yes |
| task3_hin_indicbert.yaml | indicbert-v2 | 2 | 0.1243 | 0.1875 | 0.5154 | yes |
| task3_hin_mbert.yaml | mbert-base | 0 | 0.3291 | 0.4375 | 1.0000 | yes |
| task3_hin_mbert.yaml | mbert-base | 1 | 0.2971 | 0.3750 | 1.0000 | yes |
| task3_hin_mbert.yaml | mbert-base | 2 | 0.3798 | 0.4375 | 1.0000 | yes |
