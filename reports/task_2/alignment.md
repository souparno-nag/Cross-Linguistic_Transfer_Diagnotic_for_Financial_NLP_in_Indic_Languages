# T-102b — Task 2 alignment

Method: mutual nearest neighbour on sentence-transformers/LaBSE at τ=0.82, three-way consistent · config hash `5b4e7d62f58d8c08`

| Language | Upstream rows | Aligned rows |
|---|---|---|
| hindi | 2238 | 1769 |
| bengali | 2228 | 1769 |
| telugu | 2072 | 1769 |

**1769 items** align across all three languages.

Pairwise mutual matches before requiring three-way agreement:

| Pair | Mutual matches above τ |
|---|---|
| hindi-bengali | 2029 |
| hindi-telugu | 2019 |
| bengali-telugu | 1833 |

Requiring all three pairings to agree on the same triple is what reduces these to 1769. A pair surviving two independent routes is much stronger evidence than one.

Rows that did not join the three-way set, by language: hindi 469, bengali 459, telugu 303. These are **kept** in their own splits — §4 rule 1 forbids deleting them — but they cannot form a parallel item, either because the split sizes genuinely differ or because no confident mutual match was found.

Label agreement across languages: **complete** — every aligned item carries the same label in all three languages.
