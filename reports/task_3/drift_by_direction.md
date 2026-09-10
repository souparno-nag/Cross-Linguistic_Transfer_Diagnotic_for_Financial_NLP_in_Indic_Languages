# T-109 — translation drift by direction, task 3

τ = 0.73, config `4a0ec72426e2cd9d`, 4788 MT pairs, every pair scored (T-108).

Below-τ rows carry `translation_drift` and **remain in the corpus** — §4 rule 1 makes a low-similarity pair a reported category, not garbage.

| src_lang   | tgt_lang   |   rows |   mean |   median |    p05 |    p25 |   below_tau |   below_share |
|:-----------|:-----------|-------:|-------:|---------:|-------:|-------:|------------:|--------------:|
| hin        | mal        |    532 | 0.8459 |   0.8588 | 0.7372 | 0.8073 |          24 |        0.0451 |
| tel        | mal        |    532 | 0.8514 |   0.8607 | 0.7458 | 0.8169 |          20 |        0.0376 |
| ben        | mal        |    532 | 0.8597 |   0.8702 | 0.762  | 0.8295 |          11 |        0.0207 |
| tel        | ben        |    532 | 0.8706 |   0.8812 | 0.7745 | 0.8444 |          10 |        0.0188 |
| tel        | hin        |    532 | 0.882  |   0.8905 | 0.7855 | 0.8595 |           6 |        0.0113 |
| ben        | tel        |    532 | 0.8794 |   0.8889 | 0.7769 | 0.8563 |           6 |        0.0113 |
| hin        | tel        |    532 | 0.8927 |   0.9028 | 0.8008 | 0.8697 |           4 |        0.0075 |
| ben        | hin        |    532 | 0.8908 |   0.8984 | 0.7982 | 0.8671 |           2 |        0.0038 |
| hin        | ben        |    532 | 0.8986 |   0.9083 | 0.8107 | 0.8756 |           2 |        0.0038 |

Worst direction: `hin→mal` at 4.5% below τ (median <bound method Series.median of src_lang          hin
tgt_lang          mal
rows              532
mean           0.8459
median         0.8588
p05            0.7372
p25            0.8073
below_tau          24
below_share    0.0451
Name: 4, dtype: object>).

## By gold class

| src_lang   | tgt_lang   | gold_class                  |   rows |   median |   below_share |
|:-----------|:-----------|:----------------------------|-------:|---------:|--------------:|
| ben        | hin        | climate change              |     92 |   0.8983 |        0      |
| ben        | hin        | corporate behavior          |     30 |   0.9034 |        0      |
| ben        | hin        | corporate governance        |     91 |   0.8891 |        0.011  |
| ben        | hin        | environmental opportunities |     72 |   0.896  |        0      |
| ben        | hin        | human capital               |     37 |   0.8943 |        0      |
| ben        | hin        | natural capital             |     50 |   0.8988 |        0      |
| ben        | hin        | pollution waste             |     44 |   0.8962 |        0.0227 |
| ben        | hin        | product liability           |     68 |   0.9053 |        0      |
| ben        | hin        | social opportunities        |     27 |   0.9111 |        0      |
| ben        | hin        | stake holder opposition     |     21 |   0.9058 |        0      |
| ben        | mal        | climate change              |     92 |   0.8673 |        0.0326 |
| ben        | mal        | corporate behavior          |     30 |   0.8773 |        0      |
| ben        | mal        | corporate governance        |     91 |   0.8689 |        0.022  |
| ben        | mal        | environmental opportunities |     72 |   0.8542 |        0.0556 |
| ben        | mal        | human capital               |     37 |   0.8766 |        0.027  |
| ben        | mal        | natural capital             |     50 |   0.8711 |        0      |
| ben        | mal        | pollution waste             |     44 |   0.8688 |        0.0227 |
| ben        | mal        | product liability           |     68 |   0.8752 |        0      |
| ben        | mal        | social opportunities        |     27 |   0.8754 |        0      |
| ben        | mal        | stake holder opposition     |     21 |   0.8774 |        0      |
| ben        | tel        | climate change              |     92 |   0.8795 |        0      |
| ben        | tel        | corporate behavior          |     30 |   0.8824 |        0      |
| ben        | tel        | corporate governance        |     91 |   0.882  |        0.033  |
| ben        | tel        | environmental opportunities |     72 |   0.8872 |        0      |
| ben        | tel        | human capital               |     37 |   0.8851 |        0      |
| ben        | tel        | natural capital             |     50 |   0.8913 |        0.06   |
| ben        | tel        | pollution waste             |     44 |   0.9063 |        0      |
| ben        | tel        | product liability           |     68 |   0.8949 |        0      |
| ben        | tel        | social opportunities        |     27 |   0.8966 |        0      |
| ben        | tel        | stake holder opposition     |     21 |   0.8849 |        0      |
| hin        | ben        | climate change              |     92 |   0.909  |        0      |
| hin        | ben        | corporate behavior          |     30 |   0.899  |        0      |
| hin        | ben        | corporate governance        |     91 |   0.9065 |        0.022  |
| hin        | ben        | environmental opportunities |     72 |   0.8947 |        0      |
| hin        | ben        | human capital               |     37 |   0.9119 |        0      |
| hin        | ben        | natural capital             |     50 |   0.9083 |        0      |
| hin        | ben        | pollution waste             |     44 |   0.9079 |        0      |
| hin        | ben        | product liability           |     68 |   0.9043 |        0      |
| hin        | ben        | social opportunities        |     27 |   0.9247 |        0      |
| hin        | ben        | stake holder opposition     |     21 |   0.9142 |        0      |
| hin        | mal        | climate change              |     92 |   0.8425 |        0.0652 |
| hin        | mal        | corporate behavior          |     30 |   0.8692 |        0      |
| hin        | mal        | corporate governance        |     91 |   0.8538 |        0.0659 |
| hin        | mal        | environmental opportunities |     72 |   0.8461 |        0.0139 |
| hin        | mal        | human capital               |     37 |   0.8546 |        0.0811 |
| hin        | mal        | natural capital             |     50 |   0.8459 |        0.08   |
| hin        | mal        | pollution waste             |     44 |   0.8678 |        0.0455 |
| hin        | mal        | product liability           |     68 |   0.8709 |        0.0294 |
| hin        | mal        | social opportunities        |     27 |   0.8695 |        0      |
| hin        | mal        | stake holder opposition     |     21 |   0.8784 |        0      |
| hin        | tel        | climate change              |     92 |   0.8964 |        0.0217 |
| hin        | tel        | corporate behavior          |     30 |   0.8893 |        0      |
| hin        | tel        | corporate governance        |     91 |   0.9029 |        0.022  |
| hin        | tel        | environmental opportunities |     72 |   0.9062 |        0      |
| hin        | tel        | human capital               |     37 |   0.9082 |        0      |
| hin        | tel        | natural capital             |     50 |   0.8869 |        0      |
| hin        | tel        | pollution waste             |     44 |   0.9025 |        0      |
| hin        | tel        | product liability           |     68 |   0.9043 |        0      |
| hin        | tel        | social opportunities        |     27 |   0.9079 |        0      |
| hin        | tel        | stake holder opposition     |     21 |   0.9214 |        0      |
| tel        | ben        | climate change              |     92 |   0.8764 |        0.0217 |
| tel        | ben        | corporate behavior          |     30 |   0.8865 |        0.0333 |
| tel        | ben        | corporate governance        |     91 |   0.8721 |        0.033  |
| tel        | ben        | environmental opportunities |     72 |   0.8853 |        0      |
| tel        | ben        | human capital               |     37 |   0.8885 |        0      |
| tel        | ben        | natural capital             |     50 |   0.887  |        0.04   |
| tel        | ben        | pollution waste             |     44 |   0.8756 |        0      |
| tel        | ben        | product liability           |     68 |   0.8846 |        0.0147 |
| tel        | ben        | social opportunities        |     27 |   0.8917 |        0      |
| tel        | ben        | stake holder opposition     |     21 |   0.8867 |        0.0476 |
| tel        | hin        | climate change              |     92 |   0.8834 |        0.0217 |
| tel        | hin        | corporate behavior          |     30 |   0.8991 |        0      |
| tel        | hin        | corporate governance        |     91 |   0.8874 |        0.022  |
| tel        | hin        | environmental opportunities |     72 |   0.902  |        0      |
| tel        | hin        | human capital               |     37 |   0.8917 |        0      |
| tel        | hin        | natural capital             |     50 |   0.8939 |        0.04   |
| tel        | hin        | pollution waste             |     44 |   0.8897 |        0      |
| tel        | hin        | product liability           |     68 |   0.8908 |        0      |
| tel        | hin        | social opportunities        |     27 |   0.8877 |        0      |
| tel        | hin        | stake holder opposition     |     21 |   0.8988 |        0      |
| tel        | mal        | climate change              |     92 |   0.8589 |        0.0217 |
| tel        | mal        | corporate behavior          |     30 |   0.8636 |        0.0333 |
| tel        | mal        | corporate governance        |     91 |   0.8485 |        0.0769 |
| tel        | mal        | environmental opportunities |     72 |   0.8753 |        0.0139 |
| tel        | mal        | human capital               |     37 |   0.8591 |        0      |
| tel        | mal        | natural capital             |     50 |   0.8704 |        0.04   |
| tel        | mal        | pollution waste             |     44 |   0.882  |        0.0227 |
| tel        | mal        | product liability           |     68 |   0.8571 |        0.0588 |
| tel        | mal        | social opportunities        |     27 |   0.851  |        0.037  |
| tel        | mal        | stake holder opposition     |     21 |   0.8754 |        0.0476 |
