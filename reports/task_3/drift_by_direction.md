# T-109 — translation drift by direction, task 3

τ = 0.82, config `d93963998a4be524`, 4788 MT pairs, every pair scored (T-108).

Below-τ rows carry `translation_drift` and **remain in the corpus** — §4 rule 1 makes a low-similarity pair a reported category, not garbage.

| src_lang   | tgt_lang   |   rows |   mean |   median |    p05 |    p25 |   below_tau |   below_share |
|:-----------|:-----------|-------:|-------:|---------:|-------:|-------:|------------:|--------------:|
| hin        | mal        |    532 | 0.8459 |   0.8588 | 0.7372 | 0.8073 |         158 |        0.297  |
| tel        | mal        |    532 | 0.8514 |   0.8607 | 0.7458 | 0.8169 |         139 |        0.2613 |
| ben        | mal        |    532 | 0.8597 |   0.8702 | 0.762  | 0.8295 |         113 |        0.2124 |
| tel        | ben        |    532 | 0.8706 |   0.8812 | 0.7745 | 0.8444 |          86 |        0.1617 |
| ben        | tel        |    532 | 0.8794 |   0.8889 | 0.7769 | 0.8563 |          66 |        0.1241 |
| tel        | hin        |    532 | 0.882  |   0.8905 | 0.7855 | 0.8595 |          54 |        0.1015 |
| hin        | tel        |    532 | 0.8927 |   0.9028 | 0.8008 | 0.8697 |          45 |        0.0846 |
| ben        | hin        |    532 | 0.8908 |   0.8984 | 0.7982 | 0.8671 |          41 |        0.0771 |
| hin        | ben        |    532 | 0.8986 |   0.9083 | 0.8107 | 0.8756 |          38 |        0.0714 |

Worst direction: `hin→mal` at 29.7% below τ (median <bound method Series.median of src_lang          hin
tgt_lang          mal
rows              532
mean           0.8459
median         0.8588
p05            0.7372
p25            0.8073
below_tau         158
below_share     0.297
Name: 4, dtype: object>).

## By gold class

| src_lang   | tgt_lang   | gold_class                  |   rows |   median |   below_share |
|:-----------|:-----------|:----------------------------|-------:|---------:|--------------:|
| ben        | hin        | climate change              |     92 |   0.8983 |        0.1087 |
| ben        | hin        | corporate behavior          |     30 |   0.9034 |        0.0333 |
| ben        | hin        | corporate governance        |     91 |   0.8891 |        0.0769 |
| ben        | hin        | environmental opportunities |     72 |   0.896  |        0.0417 |
| ben        | hin        | human capital               |     37 |   0.8943 |        0.1351 |
| ben        | hin        | natural capital             |     50 |   0.8988 |        0.14   |
| ben        | hin        | pollution waste             |     44 |   0.8962 |        0.0455 |
| ben        | hin        | product liability           |     68 |   0.9053 |        0.0588 |
| ben        | hin        | social opportunities        |     27 |   0.9111 |        0.037  |
| ben        | hin        | stake holder opposition     |     21 |   0.9058 |        0.0476 |
| ben        | mal        | climate change              |     92 |   0.8673 |        0.2391 |
| ben        | mal        | corporate behavior          |     30 |   0.8773 |        0.1    |
| ben        | mal        | corporate governance        |     91 |   0.8689 |        0.1978 |
| ben        | mal        | environmental opportunities |     72 |   0.8542 |        0.3056 |
| ben        | mal        | human capital               |     37 |   0.8766 |        0.2432 |
| ben        | mal        | natural capital             |     50 |   0.8711 |        0.24   |
| ben        | mal        | pollution waste             |     44 |   0.8688 |        0.1818 |
| ben        | mal        | product liability           |     68 |   0.8752 |        0.1765 |
| ben        | mal        | social opportunities        |     27 |   0.8754 |        0.1481 |
| ben        | mal        | stake holder opposition     |     21 |   0.8774 |        0.1429 |
| ben        | tel        | climate change              |     92 |   0.8795 |        0.1304 |
| ben        | tel        | corporate behavior          |     30 |   0.8824 |        0.1667 |
| ben        | tel        | corporate governance        |     91 |   0.882  |        0.1319 |
| ben        | tel        | environmental opportunities |     72 |   0.8872 |        0.1111 |
| ben        | tel        | human capital               |     37 |   0.8851 |        0.1892 |
| ben        | tel        | natural capital             |     50 |   0.8913 |        0.18   |
| ben        | tel        | pollution waste             |     44 |   0.9063 |        0.1364 |
| ben        | tel        | product liability           |     68 |   0.8949 |        0.0441 |
| ben        | tel        | social opportunities        |     27 |   0.8966 |        0.0741 |
| ben        | tel        | stake holder opposition     |     21 |   0.8849 |        0.0952 |
| hin        | ben        | climate change              |     92 |   0.909  |        0.0761 |
| hin        | ben        | corporate behavior          |     30 |   0.899  |        0.0667 |
| hin        | ben        | corporate governance        |     91 |   0.9065 |        0.1099 |
| hin        | ben        | environmental opportunities |     72 |   0.8947 |        0.0694 |
| hin        | ben        | human capital               |     37 |   0.9119 |        0.027  |
| hin        | ben        | natural capital             |     50 |   0.9083 |        0.1    |
| hin        | ben        | pollution waste             |     44 |   0.9079 |        0.0909 |
| hin        | ben        | product liability           |     68 |   0.9043 |        0.0294 |
| hin        | ben        | social opportunities        |     27 |   0.9247 |        0.037  |
| hin        | ben        | stake holder opposition     |     21 |   0.9142 |        0.0476 |
| hin        | mal        | climate change              |     92 |   0.8425 |        0.4022 |
| hin        | mal        | corporate behavior          |     30 |   0.8692 |        0.2    |
| hin        | mal        | corporate governance        |     91 |   0.8538 |        0.3187 |
| hin        | mal        | environmental opportunities |     72 |   0.8461 |        0.3333 |
| hin        | mal        | human capital               |     37 |   0.8546 |        0.2973 |
| hin        | mal        | natural capital             |     50 |   0.8459 |        0.36   |
| hin        | mal        | pollution waste             |     44 |   0.8678 |        0.2727 |
| hin        | mal        | product liability           |     68 |   0.8709 |        0.2353 |
| hin        | mal        | social opportunities        |     27 |   0.8695 |        0.0741 |
| hin        | mal        | stake holder opposition     |     21 |   0.8784 |        0.1429 |
| hin        | tel        | climate change              |     92 |   0.8964 |        0.0761 |
| hin        | tel        | corporate behavior          |     30 |   0.8893 |        0.1667 |
| hin        | tel        | corporate governance        |     91 |   0.9029 |        0.0989 |
| hin        | tel        | environmental opportunities |     72 |   0.9062 |        0.0556 |
| hin        | tel        | human capital               |     37 |   0.9082 |        0.1081 |
| hin        | tel        | natural capital             |     50 |   0.8869 |        0.18   |
| hin        | tel        | pollution waste             |     44 |   0.9025 |        0.0909 |
| hin        | tel        | product liability           |     68 |   0.9043 |        0.0441 |
| hin        | tel        | social opportunities        |     27 |   0.9079 |        0      |
| hin        | tel        | stake holder opposition     |     21 |   0.9214 |        0      |
| tel        | ben        | climate change              |     92 |   0.8764 |        0.1413 |
| tel        | ben        | corporate behavior          |     30 |   0.8865 |        0.1    |
| tel        | ben        | corporate governance        |     91 |   0.8721 |        0.1978 |
| tel        | ben        | environmental opportunities |     72 |   0.8853 |        0.125  |
| tel        | ben        | human capital               |     37 |   0.8885 |        0.1081 |
| tel        | ben        | natural capital             |     50 |   0.887  |        0.24   |
| tel        | ben        | pollution waste             |     44 |   0.8756 |        0.1591 |
| tel        | ben        | product liability           |     68 |   0.8846 |        0.1618 |
| tel        | ben        | social opportunities        |     27 |   0.8917 |        0.1852 |
| tel        | ben        | stake holder opposition     |     21 |   0.8867 |        0.1905 |
| tel        | hin        | climate change              |     92 |   0.8834 |        0.1196 |
| tel        | hin        | corporate behavior          |     30 |   0.8991 |        0.1333 |
| tel        | hin        | corporate governance        |     91 |   0.8874 |        0.0879 |
| tel        | hin        | environmental opportunities |     72 |   0.902  |        0.0556 |
| tel        | hin        | human capital               |     37 |   0.8917 |        0.1892 |
| tel        | hin        | natural capital             |     50 |   0.8939 |        0.1    |
| tel        | hin        | pollution waste             |     44 |   0.8897 |        0.1364 |
| tel        | hin        | product liability           |     68 |   0.8908 |        0.0588 |
| tel        | hin        | social opportunities        |     27 |   0.8877 |        0.1111 |
| tel        | hin        | stake holder opposition     |     21 |   0.8988 |        0.0952 |
| tel        | mal        | climate change              |     92 |   0.8589 |        0.2935 |
| tel        | mal        | corporate behavior          |     30 |   0.8636 |        0.1333 |
| tel        | mal        | corporate governance        |     91 |   0.8485 |        0.2857 |
| tel        | mal        | environmental opportunities |     72 |   0.8753 |        0.2222 |
| tel        | mal        | human capital               |     37 |   0.8591 |        0.2432 |
| tel        | mal        | natural capital             |     50 |   0.8704 |        0.3    |
| tel        | mal        | pollution waste             |     44 |   0.882  |        0.3182 |
| tel        | mal        | product liability           |     68 |   0.8571 |        0.2794 |
| tel        | mal        | social opportunities        |     27 |   0.851  |        0.1481 |
| tel        | mal        | stake holder opposition     |     21 |   0.8754 |        0.2381 |

## More than 20% below τ

T-110 treats this as evidence that the threshold is measuring the wrong thing rather than as a quality verdict on these directions:

- `hin→mal`: 29.7% below τ
- `tel→mal`: 26.1% below τ
- `ben→mal`: 21.2% below τ
