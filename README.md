# Kaggle-Quora-insincere-questions-classification
https://www.kaggle.com/c/quora-insincere-questions-classification

Classification of questions posted in Quora as "Sincere" and "Insincere" questions.

## Code

1. Get to know the data -- summarize_word.py

2. Clean data  --  clean_csv.py

3. Individual solution

    | Model          | Code          |
    | -------------  | ------------- |
    | Binary Word    | bw.py         |
    | Count Word     | cw.py         |
    | TF-IDF         | tfidf.py      |
    | Keras Embedding| em_keras.py   |
    | Glvoe Embedding| em_glove.py   |
    | Para Embedding | em_para.py   |
    | Wiki Embedding| em_wiki.py   |
    | Google Embedding| em_gg.py   |

    Run clean_csv.py first, and then run the file listed above.

4. Ensemble solution

    | Model          | Code          |
    | -------------  | ------------- |
    | Max Vote       | max_vote.py   |
    | Ave Vote       | ave_vote.py   |
    | XG boost       | xg_vote.py    |

    Run clean_csv.py first, and then run the file listed above.

5. Best solution  --  all_in_one.py, this can be submitted to kaggle competition directly.

## Summary Report

insincere.pdf
https://github.com/JialieY/Kaggle-Quora-insincere-questions-classification/blob/master/insincere.pdf
