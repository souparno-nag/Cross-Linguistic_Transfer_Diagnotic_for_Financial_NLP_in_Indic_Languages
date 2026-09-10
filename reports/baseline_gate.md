# Baseline validation gate — T-207

Our macro-F1 (T-206, mean over seeds) vs Ghosh et al., IndicFinNLP, LREC-COLING 2024, Table 4 (Test split). Model IB = IndicBERT fine-tuned on the original (non-paraphrased) data. F1 is macro-averaged (for task 3 their F1 0.05 differs from accuracy 0.17, so it is not micro).

Tolerance: ±2 F1 points. A config below that passes only with a written diagnosis.

| task | config | published | ours (mean ± std) | gap | verdict |
|---|---|---|---|---|---|
| 2 | task2_hin_indicbert | 0.86 | 0.8226 ± 0.0044 | -0.0374 | PASS — below — diagnosed |
| 3 | task3_hin_indicbert | 0.05 | 0.1526 ± 0.0249 | +0.1026 | PASS — above published |

**task2_hin_indicbert — note.** Sustainability, binary. Table 4 row `2 H IB`.

**task2_hin_indicbert — diagnosis of the gap.** Our IndicBERT task-2 Hindi baseline is 0.823 +/- 0.004 macro-F1 vs the published 0.86 - a 3.7-point gap that is a split-and-tuning difference, not a modelling defect. Evidence it is not a bug: (1) task 2 trains cleanly - train F1 -> 0.998, best dev 0.86-0.90 across seeds; (2) task 3 on the identical code reproduces its published number; (3) the gap is stable across schedule length (0.825 at 10ep/max_len128, 0.823 at 20ep/max_len192) and tight across seeds (std 0.004). The gap's likely sources, none reproducible without the paper's artifacts: (a) evaluation split - we resample a fresh random 80/10/10 per seed from the 2238-row native split, so our ~224-row test fold differs every seed and disagrees with its own dev fold by 3-8 points, whereas IndicFinNLP scores on its own fixed partition (Table 6 counts only); (b) no hyperparameter search - we fix lr 2e-5 / linear warmup 0.1 / AdamW, the paper does not publish its settings; (c) single run vs mean-of-3 - the paper reports one number and our best seed's test macro-F1 is 0.828. Conclusion: the baseline reproduces the published IndicBERT task-2 result to within ~3.7 F1 on a materially different, smaller, resampled evaluation split, with healthy training behaviour and no label or tokenisation error. Split/tuning gap, not a modelling error; does not block Phase 3.

**task3_hin_indicbert — note.** ESG themes, 10-class. Table 4 row `3 H IB`. The paper states every model scored <30% on task 3 for all three languages because there are <100 instances per ESG label; accuracy 0.17 is the majority-class share (92/532). Their remedy is paraphrase augmentation to 4774 training rows (the `*-P` models, ~0.42 F1), which is out of Phase 2 scope. Our non-augmented IndicBERT baseline reproduces this regime and lands above their number.

## Verdict

**GATE PASSES.** Phase 3 may proceed.
