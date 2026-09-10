# Baseline validation gate — T-207

Our macro-F1 (T-206, mean over seeds) vs Ghosh et al., IndicFinNLP, LREC-COLING 2024, Table 4 (Test split). Model IB = IndicBERT fine-tuned on the original (non-paraphrased) data. F1 is macro-averaged (for task 3 their F1 0.05 differs from accuracy 0.17, so it is not micro).

Tolerance: ±2 F1 points. A config below that passes only with a written diagnosis.

| task | config | published | ours (mean ± std) | gap | verdict |
|---|---|---|---|---|---|
| 2 | task2_hin_indicbert | 0.86 | 0.8253 ± 0.0241 | -0.0347 | FAIL — below — NO diagnosis |
| 3 | task3_hin_indicbert | 0.05 | 0.1526 ± 0.0249 | +0.1026 | PASS — above published |

**task2_hin_indicbert — note.** Sustainability, binary. Table 4 row `2 H IB`.

**task3_hin_indicbert — note.** ESG themes, 10-class. Table 4 row `3 H IB`. The paper states every model scored <30% on task 3 for all three languages because there are <100 instances per ESG label; accuracy 0.17 is the majority-class share (92/532). Their remedy is paraphrase augmentation to 4774 training rows (the `*-P` models, ~0.42 F1), which is out of Phase 2 scope. Our non-augmented IndicBERT baseline reproduces this regime and lands above their number.

## Verdict

**GATE FAILS.** Resolve the FAIL rows — improve the run or write a diagnosis — before any Phase 3 work.
