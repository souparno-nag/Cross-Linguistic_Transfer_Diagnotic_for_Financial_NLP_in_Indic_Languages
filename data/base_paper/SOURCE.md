# Base paper dataset

IndicFinNLP, the dataset accompanying *"IndicFinNLP: Financial Natural
Language Processing for Indian Languages"*.

| | |
|---|---|
| Source | <https://www.kaggle.com/datasets/sohomghosh/indicfinnlp-financial-nlp-for-indian-languages> |
| Kaggle slug | `sohomghosh/indicfinnlp-financial-nlp-for-indian-languages` |
| Version | 2 |
| Retrieved (UTC) | 2026-09-03T06:33:51+00:00 |

Licence terms are in `license.txt`; upstream's column-level metadata is in
`README.md`. Both are copied verbatim from the release.

## Layout

Files are renamed on materialisation so that every task and language shares
one shape, `raw/task_<n>/<language>.xlsx`. `manifest.json` records the
upstream name and a SHA-256 for each file.

| Task | Language | Local | Upstream |
|---|---|---|---|
| 1 | bengali | `raw/task_1/bengali.xlsx` | `IndicFinNLP_data/IndicFinNLP_data/task_1/task_1_dataset_bengali_refined_submit.xlsx` |
| 1 | hindi | `raw/task_1/hindi.xlsx` | `IndicFinNLP_data/IndicFinNLP_data/task_1/task_1_dataset_hindi_refined_submit.xlsx` |
| 1 | telugu | `raw/task_1/telugu.xlsx` | `IndicFinNLP_data/IndicFinNLP_data/task_1/task_1_dataset_telugu_refined_submit.xlsx` |
| 2 | bengali | `raw/task_2/bengali.xlsx` | `IndicFinNLP_data/IndicFinNLP_data/task_2/task_2_dataset_bengali_Sustainability.xlsx` |
| 2 | hindi | `raw/task_2/hindi.xlsx` | `IndicFinNLP_data/IndicFinNLP_data/task_2/task_2_dataset_hindi_Sustainability.xlsx` |
| 2 | telugu | `raw/task_2/telugu.xlsx` | `IndicFinNLP_data/IndicFinNLP_data/task_2/task_2_dataset_telugu_Sustainability.xlsx` |
| 3 | bengali | `raw/task_3/bengali.xlsx` | `IndicFinNLP_data/IndicFinNLP_data/task_3/task_3_dataset_bengali_submit.xlsx` |
| 3 | hindi | `raw/task_3/hindi.xlsx` | `IndicFinNLP_data/IndicFinNLP_data/task_3/task_3_dataset_hindi_submit.xlsx` |
| 3 | telugu | `raw/task_3/telugu.xlsx` | `IndicFinNLP_data/IndicFinNLP_data/task_3/task_3_dataset_telugu_submit.xlsx` |

## Regenerating

Do not edit this directory by hand:

```
python -m src.download_dataset.download          # no-op if already current
python -m src.download_dataset.download --force  # re-materialise
```
