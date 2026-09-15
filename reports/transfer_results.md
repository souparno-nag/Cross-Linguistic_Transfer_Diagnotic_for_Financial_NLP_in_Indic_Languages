# Transfer results (T-308)

Gap = source (in-language, same items) − target, XTREME convention (§T-304). "blocked" means the source language's checkpoint doesn't exist yet (Bengali/Telugu baselines are deferred, CLAUDE2.md). "within seed noise" flags a gap smaller than its own 3-seed standard deviation (hard rule 5) -- not a finding as-is; it renders as "—" when only one seed is available yet, since a standard deviation over one point is undefined, not zero.

## Task 2

### Transfer matrix — native/cross-block targets

| condition | quadrant | status | n seeds | source | target | gap (mean±std) | within seed noise? |
|---|---|---|---|---|---|---|---|
| transfer_hin_to_ben | Indo-Aryan->Indo-Aryan | ok | 3 | 0.9535 | 0.5058 | 0.4477 ± 0.0716 | no |
| transfer_hin_to_tel | Indo-Aryan->Dravidian | ok | 3 | 0.9535 | 0.5009 | 0.4526 ± 0.0380 | no |
| transfer_hin_to_mal | Indo-Aryan->Dravidian | ok | 3 | 0.9535 | 0.5667 | 0.3868 ± 0.0574 | no |
| transfer_ben_to_hin | Indo-Aryan->Indo-Aryan | ok | 3 | 0.9558 | 0.6927 | 0.2631 ± 0.0530 | no |
| transfer_ben_to_tel | Indo-Aryan->Dravidian | ok | 3 | 0.9558 | 0.5518 | 0.4040 ± 0.0148 | no |
| transfer_ben_to_mal | Indo-Aryan->Dravidian | ok | 3 | 0.9558 | 0.5885 | 0.3673 ± 0.0432 | no |
| transfer_tel_to_hin | Dravidian->Indo-Aryan | ok | 3 | 0.9573 | 0.5962 | 0.3610 ± 0.1054 | no |
| transfer_tel_to_ben | Dravidian->Indo-Aryan | ok | 3 | 0.9573 | 0.6338 | 0.3234 ± 0.0471 | no |
| transfer_tel_to_mal | Dravidian->Dravidian | ok | 3 | 0.9573 | 0.5405 | 0.4167 ± 0.0050 | no |

### Transfer matrix — same-source MT targets

| condition | quadrant | status | n seeds | source | target | gap (mean±std) | within seed noise? |
|---|---|---|---|---|---|---|---|
| transfer_hin_to_ben_mt | Indo-Aryan->Indo-Aryan | ok | 3 | 0.9526 | 0.4869 | 0.4656 ± 0.0883 | no |
| transfer_hin_to_tel_mt | Indo-Aryan->Dravidian | ok | 3 | 0.9526 | 0.4881 | 0.4645 ± 0.0246 | no |
| transfer_hin_to_mal_mt | Indo-Aryan->Dravidian | ok | 3 | 0.9526 | 0.5683 | 0.3842 ± 0.0686 | no |
| transfer_ben_to_hin_mt | Indo-Aryan->Indo-Aryan | ok | 3 | 0.9578 | 0.7016 | 0.2563 ± 0.0388 | no |
| transfer_ben_to_tel_mt | Indo-Aryan->Dravidian | ok | 3 | 0.9578 | 0.5661 | 0.3917 ± 0.0176 | no |
| transfer_ben_to_mal_mt | Indo-Aryan->Dravidian | ok | 3 | 0.9578 | 0.6037 | 0.3541 ± 0.0214 | no |
| transfer_tel_to_hin_mt | Dravidian->Indo-Aryan | ok | 3 | 0.9604 | 0.6043 | 0.3561 ± 0.0916 | no |
| transfer_tel_to_ben_mt | Dravidian->Indo-Aryan | ok | 3 | 0.9604 | 0.6471 | 0.3134 ± 0.0219 | no |
| transfer_tel_to_mal_mt | Dravidian->Dravidian | ok | 3 | 0.9604 | 0.5439 | 0.4166 ± 0.0042 | no |

### Quadrant summary

| quadrant | n cells (of 3) | mean gap | cells |
|---|---|---|---|
| Indo-Aryan->Indo-Aryan | 2 | 0.3554 | transfer_hin_to_ben, transfer_ben_to_hin |
| Indo-Aryan->Dravidian | 4 | 0.4027 | transfer_hin_to_tel, transfer_hin_to_mal, transfer_ben_to_tel, transfer_ben_to_mal |
| Dravidian->Indo-Aryan | 2 | 0.3422 | transfer_tel_to_hin, transfer_tel_to_ben |
| Dravidian->Dravidian | 1 | 0.4167 | transfer_tel_to_mal |

### Translationese comparison — hin

| block | origin | src_lang | n seeds | macro-F1 (mean±std) | Δ vs native |
|---|---|---|---|---|---|
| B | mt | ben | 3 | 0.8848 ± 0.0102 | -0.0688 |
| H | native | nan | 3 | 0.9536 ± 0.0157 | 0.0000 |
| T | mt | tel | 3 | 0.8962 ± 0.0189 | -0.0574 |

### Translationese comparison — ben

| block | origin | src_lang | n seeds | macro-F1 (mean±std) | Δ vs native |
|---|---|---|---|---|---|
| B | native | nan | 3 | 0.9533 ± 0.0165 | 0.0000 |
| H | mt | hin | 3 | 0.8914 ± 0.0102 | -0.0618 |
| T | mt | tel | 3 | 0.8791 ± 0.0167 | -0.0742 |

### Translationese comparison — tel

| block | origin | src_lang | n seeds | macro-F1 (mean±std) | Δ vs native |
|---|---|---|---|---|---|
| B | mt | ben | 3 | 0.8837 ± 0.0051 | -0.0784 |
| H | mt | hin | 3 | 0.9006 ± 0.0072 | -0.0615 |
| T | native | nan | 3 | 0.9621 ± 0.0034 | 0.0000 |

## Task 3

### Transfer matrix — native/cross-block targets

| condition | quadrant | status | n seeds | source | target | gap (mean±std) | within seed noise? |
|---|---|---|---|---|---|---|---|
| transfer_hin_to_ben | Indo-Aryan->Indo-Aryan | ok | 3 | 0.5958 | 0.0862 | 0.5096 ± 0.1676 | no |
| transfer_hin_to_tel | Indo-Aryan->Dravidian | ok | 3 | 0.5958 | 0.0817 | 0.5141 ± 0.1701 | no |
| transfer_hin_to_mal | Indo-Aryan->Dravidian | ok | 3 | 0.5958 | 0.0923 | 0.5035 ± 0.1843 | no |
| transfer_ben_to_hin | Indo-Aryan->Indo-Aryan | ok | 3 | 0.4175 | 0.0596 | 0.3579 ± 0.3083 | no |
| transfer_ben_to_tel | Indo-Aryan->Dravidian | ok | 3 | 0.4175 | 0.0739 | 0.3436 ± 0.2934 | no |
| transfer_ben_to_mal | Indo-Aryan->Dravidian | ok | 3 | 0.4175 | 0.0726 | 0.3449 ± 0.3055 | no |
| transfer_tel_to_hin | Dravidian->Indo-Aryan | ok | 3 | 0.2548 | 0.0811 | 0.1737 ± 0.2634 | yes |
| transfer_tel_to_ben | Dravidian->Indo-Aryan | ok | 3 | 0.2548 | 0.0822 | 0.1726 ± 0.2644 | yes |
| transfer_tel_to_mal | Dravidian->Dravidian | ok | 3 | 0.2548 | 0.0910 | 0.1638 ± 0.2533 | yes |

### Transfer matrix — same-source MT targets

| condition | quadrant | status | n seeds | source | target | gap (mean±std) | within seed noise? |
|---|---|---|---|---|---|---|---|
| transfer_hin_to_ben_mt | Indo-Aryan->Indo-Aryan | ok | 3 | 0.5958 | 0.0955 | 0.5003 ± 0.1671 | no |
| transfer_hin_to_tel_mt | Indo-Aryan->Dravidian | ok | 3 | 0.5958 | 0.0843 | 0.5115 ± 0.1742 | no |
| transfer_hin_to_mal_mt | Indo-Aryan->Dravidian | ok | 3 | 0.5958 | 0.0803 | 0.5155 ± 0.1887 | no |
| transfer_ben_to_hin_mt | Indo-Aryan->Indo-Aryan | ok | 3 | 0.4175 | 0.0618 | 0.3557 ± 0.3038 | no |
| transfer_ben_to_tel_mt | Indo-Aryan->Dravidian | ok | 3 | 0.4175 | 0.0631 | 0.3543 ± 0.3021 | no |
| transfer_ben_to_mal_mt | Indo-Aryan->Dravidian | ok | 3 | 0.4175 | 0.0737 | 0.3438 ± 0.3033 | no |
| transfer_tel_to_hin_mt | Dravidian->Indo-Aryan | ok | 3 | 0.2548 | 0.0858 | 0.1690 ± 0.2525 | yes |
| transfer_tel_to_ben_mt | Dravidian->Indo-Aryan | ok | 3 | 0.2548 | 0.0756 | 0.1792 ± 0.2646 | yes |
| transfer_tel_to_mal_mt | Dravidian->Dravidian | ok | 3 | 0.2548 | 0.0868 | 0.1680 ± 0.2577 | yes |

### Quadrant summary

| quadrant | n cells (of 3) | mean gap | cells |
|---|---|---|---|
| Indo-Aryan->Indo-Aryan | 2 | 0.4338 | transfer_hin_to_ben, transfer_ben_to_hin |
| Indo-Aryan->Dravidian | 4 | 0.4265 | transfer_hin_to_tel, transfer_hin_to_mal, transfer_ben_to_tel, transfer_ben_to_mal |
| Dravidian->Indo-Aryan | 2 | 0.1732 | transfer_tel_to_hin, transfer_tel_to_ben |
| Dravidian->Dravidian | 1 | 0.1638 | transfer_tel_to_mal |

### Translationese comparison — hin

| block | origin | src_lang | n seeds | macro-F1 (mean±std) | Δ vs native |
|---|---|---|---|---|---|
| B | mt | ben | 3 | 0.3324 ± 0.0797 | -0.2612 |
| H | native | nan | 3 | 0.5936 ± 0.1863 | 0.0000 |
| T | mt | tel | 3 | 0.3275 ± 0.0621 | -0.2661 |

### Translationese comparison — ben

| block | origin | src_lang | n seeds | macro-F1 (mean±std) | Δ vs native |
|---|---|---|---|---|---|
| B | native | nan | 3 | 0.4292 ± 0.3136 | 0.0000 |
| H | mt | hin | 3 | 0.2040 ± 0.1240 | -0.2251 |
| T | mt | tel | 3 | 0.2035 ± 0.1239 | -0.2256 |

### Translationese comparison — tel

| block | origin | src_lang | n seeds | macro-F1 (mean±std) | Δ vs native |
|---|---|---|---|---|---|
| B | mt | ben | 3 | 0.1488 ± 0.1130 | -0.1004 |
| H | mt | hin | 3 | 0.1614 ± 0.1407 | -0.0878 |
| T | native | nan | 3 | 0.2492 ± 0.2765 | 0.0000 |

