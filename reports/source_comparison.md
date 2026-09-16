# Source-selection comparison — T-408

All source blocks × all encoders × all targets. The gap is `source − target` macro-F1 on the **same items** (T-304's paired bootstrap), so a lower gap is a better transfer source and a source that simply starts higher is not flattered by where it started.

Every number is read back out of T-305's prediction logs through T-308's own summariser. Nothing here is recomputed by a second route, because a summary that derives its own numbers can disagree with the report it summarises, silently.

## Seeds excluded, and why this is the one place that is allowed

A model that never left its initialisation scores equally badly in its own language and in every other, so its transfer gap is near **zero** and it ranks as the *best* source in the table. That is not a subtle distortion: it inverted task 3's Bengali comparison at T-404. These seeds are therefore left out of the means below and named here rather than hidden (`src/convergence.py`).

| run | excluded seeds |
|---|---|
| `task3_ben_indicbert` | [0] |
| `task3_tel_indicbert` | [0, 2] |

**`xlm-r-base` is absent from every table below.** VRAM-blocked: a full fine-tune needs 4.14 GiB of weights, gradients and AdamW state before any activation, against 3.68 GiB of card (CLAUDE5.md T-501). Not a missing run — an impossible one on this hardware.

## Task 2

### Transfer gap by source and target

| encoder | source | target | same family | n seeds | gap (mean±std) | source | target score |
|---|---|---|---|---|---|---|---|
| indicbert-v2 | ben | hin | yes | 3 | 0.2631 ± 0.0530 | 0.9558 | 0.6927 |
| indicbert-v2 | ben | mal | no | 3 | 0.3673 ± 0.0432 | 0.9558 | 0.5885 |
| indicbert-v2 | ben | tel | no | 3 | 0.4040 ± 0.0148 | 0.9558 | 0.5518 |
| indicbert-v2 | hin | ben | yes | 3 | 0.4477 ± 0.0716 | 0.9535 | 0.5058 |
| indicbert-v2 | hin | mal | no | 3 | 0.3868 ± 0.0574 | 0.9535 | 0.5667 |
| indicbert-v2 | hin | tel | no | 3 | 0.4526 ± 0.0380 | 0.9535 | 0.5009 |
| indicbert-v2 | tel | ben | no | 3 | 0.3234 ± 0.0471 | 0.9573 | 0.6338 |
| indicbert-v2 | tel | hin | no | 3 | 0.3610 ± 0.1054 | 0.9573 | 0.5962 |
| indicbert-v2 | tel | mal | yes | 3 | 0.4167 ± 0.0050 | 0.9573 | 0.5405 |
| mbert-base | ben | hin | yes | 3 | 0.1052 ± 0.0232 | 0.9752 | 0.8699 |
| mbert-base | ben | mal | no | 3 | 0.1789 ± 0.0277 | 0.9752 | 0.7963 |
| mbert-base | ben | tel | no | 3 | 0.1762 ± 0.0099 | 0.9752 | 0.7990 |
| mbert-base | hin | ben | yes | 3 | 0.1415 ± 0.0103 | 0.9747 | 0.8333 |
| mbert-base | hin | mal | no | 3 | 0.2090 ± 0.0071 | 0.9747 | 0.7657 |
| mbert-base | hin | tel | no | 3 | 0.2364 ± 0.0555 | 0.9747 | 0.7383 |
| mbert-base | tel | ben | no | 3 | 0.1960 ± 0.0580 | 0.9710 | 0.7750 |
| mbert-base | tel | hin | no | 3 | 0.1593 ± 0.0597 | 0.9710 | 0.8117 |
| mbert-base | tel | mal | yes | 3 | 0.1671 ± 0.0557 | 0.9710 | 0.8039 |

### Which source is the best overall?

| encoder | source | n targets | mean gap | best into | worst into |
|---|---|---|---|---|---|
| indicbert-v2 | **ben** | 3 | 0.3448 | hin | tel |
| indicbert-v2 | **tel** | 3 | 0.3671 | ben | mal |
| indicbert-v2 | **hin** | 3 | 0.4291 | mal | tel |
| mbert-base | **ben** | 3 | 0.1534 | hin | mal |
| mbert-base | **tel** | 3 | 0.1742 | hin | ben |
| mbert-base | **hin** | 3 | 0.1956 | ben | tel |

### Best source per target — does typological proximity predict it?

| encoder | target | best source | ranking (source: gap) | same-family source available | proximity predicts? |
|---|---|---|---|---|---|
| indicbert-v2 | ben | **tel** | tel: 0.3234, hin: 0.4477 | yes | **no** |
| indicbert-v2 | hin | **ben** | ben: 0.2631, tel: 0.3610 | yes | **yes** |
| indicbert-v2 | mal | **ben** | ben: 0.3673, hin: 0.3868, tel: 0.4167 | yes | **no** |
| indicbert-v2 | tel | **ben** | ben: 0.4040, hin: 0.4526 | no | n/a |
| mbert-base | ben | **hin** | hin: 0.1415, tel: 0.1960 | yes | **yes** |
| mbert-base | hin | **ben** | ben: 0.1052, tel: 0.1593 | yes | **yes** |
| mbert-base | mal | **tel** | tel: 0.1671, ben: 0.1789, hin: 0.2090 | yes | **yes** |
| mbert-base | tel | **ben** | ben: 0.1762, hin: 0.2364 | no | n/a |

### Is transfer symmetric?

A difference smaller than the two cells' combined seed spread is not a finding (`hard rule 5`, applied to the comparison rather than to either cell).

| encoder | pair | forward | reverse | difference | harder direction | beats noise? |
|---|---|---|---|---|---|---|
| indicbert-v2 | ben<->hin | ben->hin 0.2631 | hin->ben 0.4477 | 0.1846 | **hin->ben** | yes |
| indicbert-v2 | ben<->tel | ben->tel 0.4040 | tel->ben 0.3234 | 0.0806 | **ben->tel** | yes |
| indicbert-v2 | hin<->tel | hin->tel 0.4526 | tel->hin 0.3610 | 0.0916 | **hin->tel** | no |
| mbert-base | ben<->hin | ben->hin 0.1052 | hin->ben 0.1415 | 0.0362 | **hin->ben** | yes |
| mbert-base | ben<->tel | ben->tel 0.1762 | tel->ben 0.1960 | 0.0198 | **tel->ben** | no |
| mbert-base | hin<->tel | hin->tel 0.2364 | tel->hin 0.1593 | 0.0771 | **hin->tel** | no |

### Does the cost sit in the target language?

Pooled over source family, which is the comparison that separates "this target is hard" from "this pairing is distant". Phase 1's T-111 found the same shape in the machine translation itself.

| encoder | target family | n cells | mean gap |
|---|---|---|---|
| indicbert-v2 | Dravidian | 5 | 0.4055 |
| indicbert-v2 | Indo-Aryan | 4 | 0.3488 |
| mbert-base | Dravidian | 5 | 0.1935 |
| mbert-base | Indo-Aryan | 4 | 0.1505 |

## Task 3

### Transfer gap by source and target

| encoder | source | target | same family | n seeds | gap (mean±std) | source | target score |
|---|---|---|---|---|---|---|---|
| indicbert-v2 | ben | hin | yes | 2 | 0.5230 ± 0.1626 | 0.5899 | 0.0669 |
| indicbert-v2 | ben | mal | no | 2 | 0.5149 ± 0.1145 | 0.5899 | 0.0750 |
| indicbert-v2 | ben | tel | no | 2 | 0.5042 ± 0.1321 | 0.5899 | 0.0858 |
| indicbert-v2 | hin | ben | yes | 3 | 0.5096 ± 0.1676 | 0.5958 | 0.0862 |
| indicbert-v2 | hin | mal | no | 3 | 0.5035 ± 0.1843 | 0.5958 | 0.0923 |
| indicbert-v2 | hin | tel | no | 3 | 0.5141 ± 0.1701 | 0.5958 | 0.0817 |
| indicbert-v2 | tel | ben | no | 1 | 0.4777 ± 0.0000 | 0.5851 | 0.1075 |
| indicbert-v2 | tel | hin | no | 1 | 0.4778 ± 0.0000 | 0.5851 | 0.1073 |
| indicbert-v2 | tel | mal | yes | 1 | 0.4560 ± 0.0000 | 0.5851 | 0.1291 |
| mbert-base | ben | hin | yes | 3 | 0.4938 ± 0.0038 | 0.8099 | 0.3161 |
| mbert-base | ben | mal | no | 3 | 0.5379 ± 0.0314 | 0.8099 | 0.2719 |
| mbert-base | ben | tel | no | 3 | 0.5226 ± 0.0186 | 0.8099 | 0.2873 |
| mbert-base | hin | ben | yes | 3 | 0.5155 ± 0.0490 | 0.8052 | 0.2897 |
| mbert-base | hin | mal | no | 3 | 0.5611 ± 0.0499 | 0.8052 | 0.2441 |
| mbert-base | hin | tel | no | 3 | 0.5160 ± 0.0725 | 0.8052 | 0.2892 |
| mbert-base | tel | ben | no | 3 | 0.5240 ± 0.0336 | 0.7839 | 0.2599 |
| mbert-base | tel | hin | no | 3 | 0.4888 ± 0.0253 | 0.7839 | 0.2950 |
| mbert-base | tel | mal | yes | 3 | 0.5284 ± 0.0537 | 0.7839 | 0.2555 |

### Which source is the best overall?

| encoder | source | n targets | mean gap | best into | worst into |
|---|---|---|---|---|---|
| indicbert-v2 | **tel** | 3 | 0.4705 | mal | hin |
| indicbert-v2 | **hin** | 3 | 0.5091 | mal | tel |
| indicbert-v2 | **ben** | 3 | 0.5140 | tel | hin |
| mbert-base | **tel** | 3 | 0.5137 | hin | mal |
| mbert-base | **ben** | 3 | 0.5181 | hin | mal |
| mbert-base | **hin** | 3 | 0.5308 | ben | mal |

### Best source per target — does typological proximity predict it?

| encoder | target | best source | ranking (source: gap) | same-family source available | proximity predicts? |
|---|---|---|---|---|---|
| indicbert-v2 | ben | **tel** | tel: 0.4777, hin: 0.5096 | yes | **no** |
| indicbert-v2 | hin | **tel** | tel: 0.4778, ben: 0.5230 | yes | **no** |
| indicbert-v2 | mal | **tel** | tel: 0.4560, hin: 0.5035, ben: 0.5149 | yes | **yes** |
| indicbert-v2 | tel | **ben** | ben: 0.5042, hin: 0.5141 | no | n/a |
| mbert-base | ben | **hin** | hin: 0.5155, tel: 0.5240 | yes | **yes** |
| mbert-base | hin | **tel** | tel: 0.4888, ben: 0.4938 | yes | **no** |
| mbert-base | mal | **tel** | tel: 0.5284, ben: 0.5379, hin: 0.5611 | yes | **yes** |
| mbert-base | tel | **hin** | hin: 0.5160, ben: 0.5226 | no | n/a |

### Is transfer symmetric?

A difference smaller than the two cells' combined seed spread is not a finding (`hard rule 5`, applied to the comparison rather than to either cell).

| encoder | pair | forward | reverse | difference | harder direction | beats noise? |
|---|---|---|---|---|---|---|
| indicbert-v2 | ben<->hin | ben->hin 0.5230 | hin->ben 0.5096 | 0.0134 | **ben->hin** | no |
| indicbert-v2 | ben<->tel | ben->tel 0.5042 | tel->ben 0.4777 | 0.0265 | **ben->tel** | no |
| indicbert-v2 | hin<->tel | hin->tel 0.5141 | tel->hin 0.4778 | 0.0363 | **hin->tel** | no |
| mbert-base | ben<->hin | ben->hin 0.4938 | hin->ben 0.5155 | 0.0217 | **hin->ben** | no |
| mbert-base | ben<->tel | ben->tel 0.5226 | tel->ben 0.5240 | 0.0014 | **tel->ben** | no |
| mbert-base | hin<->tel | hin->tel 0.5160 | tel->hin 0.4888 | 0.0272 | **hin->tel** | no |

### Does the cost sit in the target language?

Pooled over source family, which is the comparison that separates "this target is hard" from "this pairing is distant". Phase 1's T-111 found the same shape in the machine translation itself.

| encoder | target family | n cells | mean gap |
|---|---|---|---|
| indicbert-v2 | Dravidian | 5 | 0.4985 |
| indicbert-v2 | Indo-Aryan | 4 | 0.4970 |
| mbert-base | Dravidian | 5 | 0.5332 |
| mbert-base | Indo-Aryan | 4 | 0.5055 |

