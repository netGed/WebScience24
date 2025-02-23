import joblib
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from webapp.backend.app.types import Tweet

tfidf_vectorizer_mixeddata = joblib.load("models/ensemble/tfidf_vectorizer_for_brf.joblib")


def generate_classification_metrics_for_ensemble(tweets: list[Tweet]):
    # convert data
    df = pd.DataFrame([vars(s) for s in tweets])
    X = df["tweet"]
    y = df["label"]

    # load model
    model = joblib.load("models/ensemble/brf_untuned_tfidf_model.joblib")

    # clean tweets before vectorization
    # todo?

    # vectorize tweets
    X_vectorized = tfidf_vectorizer_mixeddata.transform(X)

    # generate metrics
    acc = model.score(X_vectorized, y)
    precision = precision_score(y, model.predict(X_vectorized))
    recall = recall_score(y, model.predict(X_vectorized))
    f1 = f1_score(y, model.predict(X_vectorized))

    result = {
        "model_name": "Balanced Random Forest",
        "accuracy": round(acc, 2),
        "f1_score": round(f1, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
    }

    return result


# todo
def generate_classification_metrics_for_svm(tweets):
    # convert data
    df = pd.DataFrame([vars(s) for s in tweets])
    X = df["tweet"]
    y = df["label"]

    # load model
    model = joblib.load("models/ensemble/brf_untuned_tfidf_model.joblib")

    # clean tweets before vectorization
    # todo?

    # vectorize tweets
    X_vectorized = tfidf_vectorizer_mixeddata.transform(X)

    # generate metrics
    # acc = model.score(X_vectorized, y)
    # precision = precision_score(y, model.predict(X_vectorized))
    # recall = recall_score(y, model.predict(X_vectorized))
    # f1 = f1_score(y, model.predict(X_vectorized))

    # mock classification
    acc = round(random.randint(0, 99) / 100, 2)
    precision = round(random.randint(0, 99) / 100, 2)
    recall = round(random.randint(0, 99) / 100, 2)
    f1 = round((2 * precision * recall) / (precision + recall))

    result = {
        "model_name": "SVM",
        "accuracy": round(acc, 2),
        "f1_score": round(f1, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
    }

    return result


# todo
def generate_classification_metrics_for_nb(tweets):
    # convert data
    df = pd.DataFrame([vars(s) for s in tweets])
    X = df["tweet"]
    y = df["label"]

    # load model
    model = joblib.load("models/ensemble/brf_untuned_tfidf_model.joblib")

    # clean tweets before vectorization
    # todo?

    # vectorize tweets
    X_vectorized = tfidf_vectorizer_mixeddata.transform(X)

    # generate metrics
    # acc = model.score(X_vectorized, y)
    # precision = precision_score(y, model.predict(X_vectorized))
    # recall = recall_score(y, model.predict(X_vectorized))
    # f1 = f1_score(y, model.predict(X_vectorized))

    # mock classification
    acc = round(random.randint(0, 99) / 100, 2)
    precision = round(random.randint(0, 99) / 100, 2)
    recall = round(random.randint(0, 99) / 100, 2)
    f1 = round((2 * precision * recall) / (precision + recall))

    result = {
        "model_name": "Naive Bayes",
        "accuracy": round(acc, 2),
        "f1_score": round(f1, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
    }

    return result


# todo
def generate_classification_metrics_for_gru(tweets):
    # convert data
    df = pd.DataFrame([vars(s) for s in tweets])
    X = df["tweet"]
    y = df["label"]

    # load model
    model = joblib.load("models/ensemble/brf_untuned_tfidf_model.joblib")

    # clean tweets before vectorization
    # todo?

    # vectorize tweets
    X_vectorized = tfidf_vectorizer_mixeddata.transform(X)

    # generate metrics
    # acc = model.score(X_vectorized, y)
    # precision = precision_score(y, model.predict(X_vectorized))
    # recall = recall_score(y, model.predict(X_vectorized))
    # f1 = f1_score(y, model.predict(X_vectorized))

    # mock classification
    acc = round(random.randint(0, 99) / 100, 2)
    precision = round(random.randint(0, 99) / 100, 2)
    recall = round(random.randint(0, 99) / 100, 2)
    f1 = round((2 * precision * recall) / (precision + recall))

    result = {
        "model_name": "RNN-GRU",
        "accuracy": round(acc, 2),
        "f1_score": round(f1, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
    }

    return result


# todo
def generate_classification_metrics_for_lstm(tweets):
    # convert data
    df = pd.DataFrame([vars(s) for s in tweets])
    X = df["tweet"]
    y = df["label"]

    # load model
    model = joblib.load("models/ensemble/brf_untuned_tfidf_model.joblib")

    # clean tweets before vectorization
    # todo?

    # vectorize tweets
    X_vectorized = tfidf_vectorizer_mixeddata.transform(X)

    # generate metrics
    # acc = model.score(X_vectorized, y)
    # precision = precision_score(y, model.predict(X_vectorized))
    # recall = recall_score(y, model.predict(X_vectorized))
    # f1 = f1_score(y, model.predict(X_vectorized))

    # mock classification
    acc = round(random.randint(0, 99) / 100, 2)
    precision = round(random.randint(0, 99) / 100, 2)
    recall = round(random.randint(0, 99) / 100, 2)
    f1 = round((2 * precision * recall) / (precision + recall))

    result = {
        "model_name": "RNN-LSTM",
        "accuracy": round(acc, 2),
        "f1_score": round(f1, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
    }

    return result


# todo
def generate_classification_metrics_for_bert(tweets):
    # convert data
    df = pd.DataFrame([vars(s) for s in tweets])
    X = df["tweet"]
    y = df["label"]

    # load model
    model = joblib.load("models/ensemble/brf_untuned_tfidf_model.joblib")

    # clean tweets before vectorization
    # todo?

    # vectorize tweets
    X_vectorized = tfidf_vectorizer_mixeddata.transform(X)

    # generate metrics
    # acc = model.score(X_vectorized, y)
    # precision = precision_score(y, model.predict(X_vectorized))
    # recall = recall_score(y, model.predict(X_vectorized))
    # f1 = f1_score(y, model.predict(X_vectorized))

    # mock classification
    acc = round(random.randint(0, 99) / 100, 2)
    precision = round(random.randint(0, 99) / 100, 2)
    recall = round(random.randint(0, 99) / 100, 2)
    f1 = round((2 * precision * recall) / (precision + recall))

    result = {
        "model_name": "BERT",
        "accuracy": round(acc, 2),
        "f1_score": round(f1, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
    }

    return result


ROBERTA_MODEL_PATH = f"models/roberta/cardiffnlp/twitter-roberta-base-sentiment"
tokenizer_roberta = AutoTokenizer.from_pretrained(ROBERTA_MODEL_PATH, map_location=torch.device('cpu'),
                                                  local_files_only=True)
model_roberta = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL_PATH, local_files_only=True)


def generate_classification_metrics_for_roberta(tweets: list[Tweet]):
    label = []
    predictions = []

    for tweet in tweets:
        label.append(tweet.label)
        encoded_input = tokenizer_roberta(tweet.tweet, return_tensors='pt')
        output = model_roberta(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        if np.argmax(scores) == 0:
            predictions.append(1)
        else:
            predictions.append(0)

    # generate metrics
    acc = accuracy_score(label, predictions)
    precision = precision_score(label, predictions)
    recall = recall_score(label, predictions)
    f1 = f1_score(label, predictions)

    result = {
        "model_name": "ROBERTA",
        "accuracy": round(acc, 2),
        "f1_score": round(f1, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
    }

    return result
