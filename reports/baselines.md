# In-language baselines — T-206

IndicBERT-v2 fine-tuned on each native Hindi split, three seeds. Numbers are on the held-out **test** fold. Compared against IndicFinNLP's published monolingual baseline in T-207.

| config | split | seeds | macro-F1 (mean ± std) | accuracy (mean ± std) |
|---|---|---|---|---|
| task2_hin_indicbert.yaml | task2/H/hin/native | 0,1,2 | 0.8226 ± 0.0044 | 0.8244 ± 0.0052 |
| task3_hin_indicbert.yaml | task3/H/hin/native | 0,1,2 | 0.1526 ± 0.0249 | 0.2250 ± 0.0331 |

## Per-seed macro-F1

| config | seed | macro-F1 | accuracy |
|---|---|---|---|
| task2_hin_indicbert.yaml | 0 | 0.8194 | 0.8214 |
| task2_hin_indicbert.yaml | 1 | 0.8209 | 0.8214 |
| task2_hin_indicbert.yaml | 2 | 0.8277 | 0.8304 |
| task3_hin_indicbert.yaml | 0 | 0.1620 | 0.2375 |
| task3_hin_indicbert.yaml | 1 | 0.1714 | 0.2500 |
| task3_hin_indicbert.yaml | 2 | 0.1243 | 0.1875 |
