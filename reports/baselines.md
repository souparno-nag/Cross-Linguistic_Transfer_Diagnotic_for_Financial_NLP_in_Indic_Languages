# In-language baselines — T-206

IndicBERT-v2 fine-tuned on each native Hindi split, three seeds. Numbers are on the held-out **test** fold. Compared against IndicFinNLP's published monolingual baseline in T-207.

| config | split | seeds | macro-F1 (mean ± std) | accuracy (mean ± std) |
|---|---|---|---|---|
| task2_hin_indicbert.yaml | task2/H/hin/native | 0,1,2 | 0.8253 ± 0.0241 | 0.8259 ± 0.0249 |
| task3_hin_indicbert.yaml | task3/H/hin/native | 0,1,2 | 0.1269 ± 0.0400 | 0.2625 ± 0.0451 |

## Per-seed macro-F1

| config | seed | macro-F1 | accuracy |
|---|---|---|---|
| task2_hin_indicbert.yaml | 0 | 0.8034 | 0.8036 |
| task2_hin_indicbert.yaml | 1 | 0.8511 | 0.8527 |
| task2_hin_indicbert.yaml | 2 | 0.8214 | 0.8214 |
| task3_hin_indicbert.yaml | 0 | 0.1723 | 0.3125 |
| task3_hin_indicbert.yaml | 1 | 0.0969 | 0.2250 |
| task3_hin_indicbert.yaml | 2 | 0.1116 | 0.2500 |
