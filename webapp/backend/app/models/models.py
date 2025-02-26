import joblib
import pickle

import numpy as np
from keras.src.saving import load_model
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer
from nltk import word_tokenize

# SVM
model_svm = ""

# RNN-LSTM
model_lstm = ""

# ENSEMBLE
tfidf_vectorizer_ensemble = joblib.load("models/ensemble/tfidf_vectorizer_for_brf.joblib")
model_ensemble = joblib.load("models/ensemble/tfidf_balancedrandomforest.joblib")

# NAIVE BAYES
w2v_vectorizer_nb = joblib.load("models/bayes/vectorizer_w2v.joblib")
model_nb_w2v = joblib.load("models/bayes/model_nb_w2v.joblib")


def vectorize_w2v(tweets, loaded_vectorizer, vector_size=300):
    x_tokenized = tweets.map(word_tokenize)

    def w2v_vector(x_tokenized, vector_size):
        vec = np.zeros(vector_size).reshape((1, vector_size))
        count = 0
        for word in x_tokenized:
            try:
                vec += loaded_vectorizer.wv[word].reshape((1, vector_size))
                count += 1
            except KeyError:

                continue
        if count != 0:
            vec /= count
        return vec

    tweets_w2v = np.zeros((len(x_tokenized), 300))
    for i in range(len(x_tokenized)):
        tweets_w2v[i, :] = w2v_vector(x_tokenized.iloc[i], 300)

    return tweets_w2v


# RNN-GRU
threshold_gru = 0.35
max_len_gru = 40
model_gru = load_model("models/gru/gru-model_mixed-dataset.keras")
with open("models/gru/tokenizer_mixed-dataset.pkl", 'rb') as f:
    tokenizer_gru = pickle.load(f)

# BERT
PATH_BERT_TUNED = f"models/bert/bert-tuned20best"
tokenizer_bert = BertTokenizer.from_pretrained(PATH_BERT_TUNED, local_files_only=True)
model_bert = AutoModelForSequenceClassification.from_pretrained(PATH_BERT_TUNED, local_files_only=True)

# ROBERTA
PATH_ROBERTA_TUNED = f"models/roberta/roberta-tuned5"
tokenizer_roberta = AutoTokenizer.from_pretrained(PATH_ROBERTA_TUNED, local_files_only=True)
model_roberta = AutoModelForSequenceClassification.from_pretrained(PATH_ROBERTA_TUNED, local_files_only=True)
