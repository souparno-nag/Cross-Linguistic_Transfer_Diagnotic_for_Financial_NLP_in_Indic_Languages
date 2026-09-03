# IndicFinNLP

Dataset mentioned in the paper, "IndicFinNLP: Financial Natural Language Processing for Indian Languages". 

## Resources

**Task-1 Metadata** <br>
------------------- <br>
| column name    | explanation                                                        | example                       |
|----------------|--------------------------------------------------------------------|-------------------------------|
| indic          | financial text in indic language                                   | যা টাকায় ১০ কোটি টাকারও বেশি। |
| number_english | number present in indic text after translating it to English       | 10                            |
| number_indic   | number present in indic text                                       | ১০                            |
| start_posn     | starting position of the number in indic text                      | 9                             |
| end_posn       | ending position of the number in indic text                        | 11                            |
| language       | indic language in which the text is present (hindi/bengali/telugu) | bengali                       |
| magnitude      | magnitude of the number                                            | 1                             |



**Task-2 Metadata** <br>
------------------- <br>
| column name    | explanation                      | example                                                         |
|----------------|----------------------------------|-----------------------------------------------------------------|
| sentence_indic | financial text in indic language | 2019 में, हवाई यात्रा हमारे अपने  कार्बन फुटप्रिंट का लगभग 38 प्रतिशत थी। |
| label          | sustainable or unsustainable     | unsustainable                                                   |
| language       | indic language                   | hindi                                                           |



**Task-3 Metadata** <br>
------------------- <br>
| column name      | explanation                  | example                                                                                 |
|------------------|------------------------------|-----------------------------------------------------------------------------------------|
|        URL       | url of the news title        | https://www.esgtoday.com/abn-amro-to-align-lending-investment-portfolios-with-net-zero/ |
| news_title_indic | news title in indic language | రుణాలను సమలేఖనం చేయడానికి, నికర సున్నాతో  పెట్టుబడి దస్త్రాలను సమలేఖనం చేయడానికి ABN AMRO                    |
| ESG_Theme        | ESG theme of the news title  | climate change                                                                          |
| language         | indic language               | telugu                                                                                  |


```bibtex 
@INPROCEEDINGS{ghosh2024,
  author={Ghosh, Sohom and Majhi, Arnab and Narayana, Aswartha and Naskar, Sudip Kumar},
  booktitle={LREC-COLING 2024}, 
  title={IndicFinNLP: Financial Natural Language Processing for Indian
Languages}, 
  month={05},
  year={2024},
  volume={},
  number={},
  pages={},
  url={}
}
```
