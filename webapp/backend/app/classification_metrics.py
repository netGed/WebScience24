import joblib
import random
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score

from webapp.backend.app.types import Tweet

tfidf_vectorizer_mixeddata = joblib.load("models/tfidf_vectorizer_mixeddata.joblib")


def generate_classification_metrics_for_ensemble(tweets: list[Tweet]):
    # convert data
    df = pd.DataFrame([vars(s) for s in tweets])
    X = df["tweet"]
    y = df["new_label"]

    # load model
    model = joblib.load("models/brf_untuned_tf.joblib")

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
    y = df["new_label"]

    # load model
    model = joblib.load("models/brf_untuned_tf.joblib")

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
    y = df["new_label"]

    # load model
    model = joblib.load("models/brf_untuned_tf.joblib")

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
    y = df["new_label"]

    # load model
    model = joblib.load("models/brf_untuned_tf.joblib")

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
    y = df["new_label"]

    # load model
    model = joblib.load("models/brf_untuned_tf.joblib")

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
    y = df["new_label"]

    # load model
    model = joblib.load("models/brf_untuned_tf.joblib")

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


# todo
def generate_classification_metrics_for_roberta(tweets):
    # convert data
    df = pd.DataFrame([vars(s) for s in tweets])
    X = df["tweet"]
    y = df["new_label"]

    # load model
    model = joblib.load("models/brf_untuned_tf.joblib")

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
        "model_name": "ROBERTA",
        "accuracy": round(acc, 2),
        "f1_score": round(f1, 2),
        "precision": round(precision, 2),
        "recall": round(recall, 2),
    }

    return result
