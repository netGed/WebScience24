import joblib
import random
import pandas as pd

tfidf_vectorizer_mixeddata = joblib.load("models/tfidf_vectorizer_mixeddata.joblib")


def predict_ensemble(tweet):
    # load model
    model = joblib.load("models/brf_untuned_tf.joblib")

    # clean tweet before vectorization
    # todo

    # vectorize tweet
    df = pd.DataFrame({'Tweet': [tweet]})
    tweet_vectorized = tfidf_vectorizer_mixeddata.transform([df['Tweet'][0]])

    # generate classification
    probs = model.predict_proba(tweet_vectorized)
    preds0 = str(probs[:, 0][0])
    preds1 = str(probs[:, 1][0])
    pred = str(model.predict(tweet_vectorized)[0])

    result = {
        "model_name": "Balanced Random Forest",
        "zero_proba": preds0,
        "one_proba": preds1,
        "label": pred
    }

    return result


# todo
def predict_svm(tweet):
    # load model

    # clean tweet before vectorization

    # vectorize tweet

    # generate classification
    preds0 = random.randint(0, 99) / 100
    preds1 = 1 - preds0
    if preds0 > preds1:
        pred = 0
    else:
        pred = 1

    result = {
        "model_name": "SVM",
        "zero_proba": preds0,
        "one_proba": preds1,
        "label": pred
    }

    return result


# todo
def predict_nb(tweet):
    # load model

    # clean tweet before vectorization

    # vectorize tweet

    # generate classification
    preds0 = random.randint(0, 99) / 100
    preds1 = 1 - preds0
    if preds0 > preds1:
        pred = 0
    else:
        pred = 1

    result = {
        "model_name": "Naive Bayes",
        "zero_proba": preds0,
        "one_proba": preds1,
        "label": pred
    }

    return result


# todo
def predict_gru(tweet):
    # load model

    # clean tweet before vectorization

    # vectorize tweet

    # generate classification
    preds0 = random.randint(0, 99) / 100
    preds1 = 1 - preds0
    if preds0 > preds1:
        pred = 0
    else:
        pred = 1

    result = {
        "model_name": "RNN-GRU",
        "zero_proba": preds0,
        "one_proba": preds1,
        "label": pred
    }

    return result


# todo
def predict_lstm(tweet):
    # load model

    # clean tweet before vectorization

    # vectorize tweet

    # generate classification
    preds0 = random.randint(0, 99) / 100
    preds1 = 1 - preds0
    if preds0 > preds1:
        pred = 0
    else:
        pred = 1

    result = {
        "model_name": "RNN-LSTM",
        "zero_proba": preds0,
        "one_proba": preds1,
        "label": pred
    }

    return result


# todo
def predict_bert(tweet):
    # load model

    # clean tweet before vectorization

    # vectorize tweet

    # generate classification
    preds0 = random.randint(0, 99) / 100
    preds1 = 1 - preds0
    if preds0 > preds1:
        pred = 0
    else:
        pred = 1

    result = {
        "model_name": "BERT",
        "zero_proba": preds0,
        "one_proba": preds1,
        "label": pred
    }

    return result


# todo
def predict_roberta(tweet):
    # load model

    # clean tweet before vectorization

    # vectorize tweet

    # generate classification
    preds0 = random.randint(0, 99) / 100
    preds1 = 1 - preds0
    if preds0 > preds1:
        pred = 0
    else:
        pred = 1

    result = {
        "model_name": "ROBERTA",
        "zero_proba": preds0,
        "one_proba": preds1,
        "label": pred
    }

    return result
