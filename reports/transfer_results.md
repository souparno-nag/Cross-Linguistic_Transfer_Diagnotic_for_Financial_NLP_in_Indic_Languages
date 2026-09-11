# Transfer results (T-308)

Gap = source (in-language, same items) − target, XTREME convention (§T-304). "blocked" means the source language's checkpoint doesn't exist yet (Bengali/Telugu baselines are deferred, CLAUDE2.md). "within seed noise" flags a gap smaller than its own 3-seed standard deviation (hard rule 5) -- not a finding as-is; it renders as "—" when only one seed is available yet, since a standard deviation over one point is undefined, not zero.

## Task 2

### Transfer matrix — native/cross-block targets

| condition | quadrant | status | n seeds | source | target | gap (mean±std) | within seed noise? |
|---|---|---|---|---|---|---|---|
| transfer_hin_to_ben | Indo-Aryan->Indo-Aryan | ok | 3 | 0.9535 | 0.5058 | 0.4477 ± 0.0716 | no |
| transfer_hin_to_tel | Indo-Aryan->Dravidian | ok | 3 | 0.9535 | 0.5009 | 0.4526 ± 0.0380 | no |
| transfer_hin_to_mal | Indo-Aryan->Dravidian | ok | 3 | 0.9535 | 0.5667 | 0.3868 ± 0.0574 | no |
| transfer_ben_to_hin | Indo-Aryan->Indo-Aryan | blocked | 0 | — | — | — | — |
| transfer_ben_to_tel | Indo-Aryan->Dravidian | blocked | 0 | — | — | — | — |
| transfer_ben_to_mal | Indo-Aryan->Dravidian | blocked | 0 | — | — | — | — |
| transfer_tel_to_hin | Dravidian->Indo-Aryan | blocked | 0 | — | — | — | — |
| transfer_tel_to_ben | Dravidian->Indo-Aryan | blocked | 0 | — | — | — | — |
| transfer_tel_to_mal | Dravidian->Dravidian | blocked | 0 | — | — | — | — |

### Transfer matrix — same-source MT targets

| condition | quadrant | status | n seeds | source | target | gap (mean±std) | within seed noise? |
|---|---|---|---|---|---|---|---|
| transfer_hin_to_ben_mt | Indo-Aryan->Indo-Aryan | ok | 3 | 0.9526 | 0.4869 | 0.4656 ± 0.0883 | no |
| transfer_hin_to_tel_mt | Indo-Aryan->Dravidian | ok | 3 | 0.9526 | 0.4881 | 0.4645 ± 0.0246 | no |
| transfer_hin_to_mal_mt | Indo-Aryan->Dravidian | ok | 3 | 0.9526 | 0.5683 | 0.3842 ± 0.0686 | no |
| transfer_ben_to_hin_mt | Indo-Aryan->Indo-Aryan | blocked | 0 | — | — | — | — |
| transfer_ben_to_tel_mt | Indo-Aryan->Dravidian | blocked | 0 | — | — | — | — |
| transfer_ben_to_mal_mt | Indo-Aryan->Dravidian | blocked | 0 | — | — | — | — |
| transfer_tel_to_hin_mt | Dravidian->Indo-Aryan | blocked | 0 | — | — | — | — |
| transfer_tel_to_ben_mt | Dravidian->Indo-Aryan | blocked | 0 | — | — | — | — |
| transfer_tel_to_mal_mt | Dravidian->Dravidian | blocked | 0 | — | — | — | — |

### Quadrant summary

| quadrant | n cells (of 3) | mean gap | cells |
|---|---|---|---|
| Indo-Aryan->Indo-Aryan | 1 | 0.4477 | transfer_hin_to_ben |
| Indo-Aryan->Dravidian | 2 | 0.4197 | transfer_hin_to_tel, transfer_hin_to_mal |
| Dravidian->Indo-Aryan | 0 | — | — |
| Dravidian->Dravidian | 0 | — | — |

### Translationese comparison — hin

| block | origin | src_lang | n seeds | macro-F1 (mean±std) | Δ vs native |
|---|---|---|---|---|---|
| B | mt | ben | 3 | 0.8848 ± 0.0102 | -0.0688 |
| H | native | nan | 3 | 0.9536 ± 0.0157 | 0.0000 |
| T | mt | tel | 3 | 0.8962 ± 0.0189 | -0.0574 |

### Translationese comparison — ben

_no `translationese_ben` prediction log on disk yet._

### Translationese comparison — tel

_no `translationese_tel` prediction log on disk yet._

## Task 3

### Transfer matrix — native/cross-block targets

| condition | quadrant | status | n seeds | source | target | gap (mean±std) | within seed noise? |
|---|---|---|---|---|---|---|---|
| transfer_hin_to_ben | Indo-Aryan->Indo-Aryan | ok | 3 | 0.5958 | 0.0862 | 0.5096 ± 0.1676 | no |
| transfer_hin_to_tel | Indo-Aryan->Dravidian | ok | 3 | 0.5958 | 0.0817 | 0.5141 ± 0.1701 | no |
| transfer_hin_to_mal | Indo-Aryan->Dravidian | ok | 3 | 0.5958 | 0.0923 | 0.5035 ± 0.1843 | no |
| transfer_ben_to_hin | Indo-Aryan->Indo-Aryan | blocked | 0 | — | — | — | — |
| transfer_ben_to_tel | Indo-Aryan->Dravidian | blocked | 0 | — | — | — | — |
| transfer_ben_to_mal | Indo-Aryan->Dravidian | blocked | 0 | — | — | — | — |
| transfer_tel_to_hin | Dravidian->Indo-Aryan | blocked | 0 | — | — | — | — |
| transfer_tel_to_ben | Dravidian->Indo-Aryan | blocked | 0 | — | — | — | — |
| transfer_tel_to_mal | Dravidian->Dravidian | blocked | 0 | — | — | — | — |

### Transfer matrix — same-source MT targets

| condition | quadrant | status | n seeds | source | target | gap (mean±std) | within seed noise? |
|---|---|---|---|---|---|---|---|
| transfer_hin_to_ben_mt | Indo-Aryan->Indo-Aryan | ok | 3 | 0.5958 | 0.0955 | 0.5003 ± 0.1671 | no |
| transfer_hin_to_tel_mt | Indo-Aryan->Dravidian | ok | 3 | 0.5958 | 0.0843 | 0.5115 ± 0.1742 | no |
| transfer_hin_to_mal_mt | Indo-Aryan->Dravidian | ok | 3 | 0.5958 | 0.0803 | 0.5155 ± 0.1887 | no |
| transfer_ben_to_hin_mt | Indo-Aryan->Indo-Aryan | blocked | 0 | — | — | — | — |
| transfer_ben_to_tel_mt | Indo-Aryan->Dravidian | blocked | 0 | — | — | — | — |
| transfer_ben_to_mal_mt | Indo-Aryan->Dravidian | blocked | 0 | — | — | — | — |
| transfer_tel_to_hin_mt | Dravidian->Indo-Aryan | blocked | 0 | — | — | — | — |
| transfer_tel_to_ben_mt | Dravidian->Indo-Aryan | blocked | 0 | — | — | — | — |
| transfer_tel_to_mal_mt | Dravidian->Dravidian | blocked | 0 | — | — | — | — |

### Quadrant summary

| quadrant | n cells (of 3) | mean gap | cells |
|---|---|---|---|
| Indo-Aryan->Indo-Aryan | 1 | 0.5096 | transfer_hin_to_ben |
| Indo-Aryan->Dravidian | 2 | 0.5088 | transfer_hin_to_tel, transfer_hin_to_mal |
| Dravidian->Indo-Aryan | 0 | — | — |
| Dravidian->Dravidian | 0 | — | — |

### Translationese comparison — hin

| block | origin | src_lang | n seeds | macro-F1 (mean±std) | Δ vs native |
|---|---|---|---|---|---|
| B | mt | ben | 3 | 0.3324 ± 0.0797 | -0.2617 |
| H | native | nan | 3 | 0.5941 ± 0.1854 | 0.0000 |
| T | mt | tel | 3 | 0.3275 ± 0.0621 | -0.2667 |

### Translationese comparison — ben

_no `translationese_ben` prediction log on disk yet._

### Translationese comparison — tel

_no `translationese_tel` prediction log on disk yet._

