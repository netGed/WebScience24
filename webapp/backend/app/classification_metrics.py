import random

import pandas as pd
from keras.src.utils import pad_sequences
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, matthews_corrcoef
from scipy.special import softmax

from webapp.backend.app.models.models import model_gru, threshold_gru, tokenizer_gru, max_len_gru, model_ensemble, \
    tfidf_vectorizer_ensemble, model_svm, model_lstm, tokenizer_bert, model_bert, tokenizer_roberta, \
    model_roberta, vectorizer_nb_tfidf, vectorizer_svm_tfidf
from webapp.backend.app.types import TweetData


def generate_classification_metrics_for_ensemble(tweets: list[TweetData]):
    # convert data
    df = pd.DataFrame([vars(s) for s in tweets])
    X = df["tweet"]
    y = df["label"]

    # load model
    model = model_ensemble

    # clean tweets before vectorization
    # todo?

    # vectorize tweets
    X_vectorized = tfidf_vectorizer_ensemble.transform(X)

    # generate metrics
    pred = model.predict(X_vectorized)
    acc = accuracy_score(y, pred)
    precision = precision_score(y, pred)
    recall = recall_score(y, pred)
    f1 = f1_score(y, pred)
    mcc = matthews_corrcoef(y, pred)

    result = {
        "model_name": "Ensemble",
        "accuracy": round(acc, 2),
        "f1_score": round(f1, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "mcc": round(mcc, 2),
    }

    return result


def generate_classification_metrics_for_svm(tweets):
    # convert data
    df = pd.DataFrame([vars(s) for s in tweets])
    X = df["tweet"]
    y = df["label"]

    # load model
    model = model_svm

    # clean tweets before vectorization
    # todo?

    # vectorize tweets
    X_vectorized = vectorizer_svm_tfidf.transform(X)

    # generate metrics
    pred = model.predict(X_vectorized)
    acc = accuracy_score(y, pred)
    precision = precision_score(y, pred)
    recall = recall_score(y, pred)
    f1 = f1_score(y, pred)
    mcc = matthews_corrcoef(y, pred)

    result = {
        "model_name": "SVM",
        "accuracy": round(acc, 2),
        "f1_score": round(f1, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "mcc": round(mcc, 2),
    }

    return result


def generate_classification_metrics_for_nb(tweets):
    # convert data
    df = pd.DataFrame([vars(s) for s in tweets])
    X = df["tweet"]
    y = df["label"]

    # load model
    model = model_ensemble

    # vectorize tweets
    df = pd.DataFrame(X)
    X_vectorized = vectorizer_nb_tfidf.transform(X)

    # generate metrics
    pred = model.predict(X_vectorized)
    acc = accuracy_score(y, pred)
    precision = precision_score(y, pred)
    recall = recall_score(y, pred)
    f1 = f1_score(y, pred)
    mcc = matthews_corrcoef(y, pred)

    result = {
        "model_name": "Naive Bayes",
        "accuracy": round(acc, 2),
        "f1_score": round(f1, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "mcc": round(mcc, 2),
    }

    return result


def generate_classification_metrics_for_gru(tweets):
    # convert data
    df = pd.DataFrame([vars(s) for s in tweets])
    X = df["tweet"]
    y = df["label"]

    # load model
    model = model_gru

    # vectorize tweets
    X_seq = tokenizer_gru.texts_to_sequences(X)
    X_tokenized = pad_sequences(X_seq, padding='post', maxlen=max_len_gru)

    # generate metrics
    pred = (model.predict(X_tokenized) > threshold_gru).astype(int)
    acc = accuracy_score(y, pred)
    precision = precision_score(y, pred)
    recall = recall_score(y, pred)
    f1 = f1_score(y, pred)
    mcc = matthews_corrcoef(y, pred)

    result = {
        "model_name": "RNN-GRU",
        "accuracy": round(acc, 2),
        "f1_score": round(f1, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "mcc": round(mcc, 2),
    }

    return result


# todo
def generate_classification_metrics_for_lstm(tweets):
    # convert data
    df = pd.DataFrame([vars(s) for s in tweets])
    X = df["tweet"]
    y = df["label"]

    # load model
    model = model_lstm

    # vectorize tweets
    X_vectorized = tfidf_vectorizer_ensemble.transform(X)

    # generate metrics
    # acc = model.score(X_vectorized, y)
    # precision = precision_score(y, model.predict(X_vectorized))
    # recall = recall_score(y, model.predict(X_vectorized))
    # f1 = f1_score(y, model.predict(X_vectorized))

    # mock classification
    acc = random.randint(0, 99) / 100
    precision = random.randint(0, 99) / 100
    recall = random.randint(0, 99) / 100
    f1 = (2 * precision * recall) / (precision + recall)
    mcc = random.randint(-99, 99) / 100

    result = {
        "model_name": "RNN-LSTM",
        "accuracy": round(acc, 2),
        "f1_score": round(f1, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "mcc": round(mcc, 2),
    }

    return result


def generate_classification_metrics_for_bert(tweets):
    label = []
    predictions = []

    for tweet in tweets:
        label.append(tweet.label)
        encoded_input = tokenizer_bert(tweet.tweet, return_tensors='pt')
        output = model_bert(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        if scores[0] > 0.5:
            predictions.append(0)
        else:
            predictions.append(1)

    # generate metrics
    acc = accuracy_score(label, predictions)
    precision = precision_score(label, predictions)
    recall = recall_score(label, predictions)
    f1 = f1_score(label, predictions)
    mcc = matthews_corrcoef(label, predictions)

    result = {
        "model_name": "BERT",
        "accuracy": round(acc, 2),
        "f1_score": round(f1, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "mcc": round(mcc, 2),
    }

    return result


def generate_classification_metrics_for_roberta(tweets: list[TweetData]):
    label = []
    predictions = []

    for tweet in tweets:
        label.append(tweet.label)
        encoded_input = tokenizer_roberta(tweet.tweet, return_tensors='pt')
        output = model_roberta(**encoded_input)
        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        if scores[0] > 0.5:
            predictions.append(0)
        else:
            predictions.append(1)

    # generate metrics
    acc = accuracy_score(label, predictions)
    precision = precision_score(label, predictions)
    recall = recall_score(label, predictions)
    f1 = f1_score(label, predictions)
    mcc = matthews_corrcoef(label, predictions)

    result = {
        "model_name": "ROBERTA",
        "accuracy": round(acc, 2),
        "f1_score": round(f1, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
        "mcc": round(mcc, 2),
    }

    return result
