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
    preds0 = round(probs[:, 0][0], 2)
    preds1 = round(probs[:, 1][0], 2)
    pred = model.predict(tweet_vectorized)[0]

    result = {
        "model_name": "Balanced Random Forest",
        "zero_proba": str(preds0),
        "one_proba": str(preds1),
        "label": str(pred),
    }

    return result


# todo
def predict_svm(tweet):
    # load model

    # clean tweet before vectorization

    # vectorize tweet

    # generate classification
    # probs = model.predict_proba(tweet_vectorized)
    # preds0 = round(probs[:, 0][0], 2)
    # preds1 = round(probs[:, 1][0], 2)
    # pred = model.predict(tweet_vectorized)[0]

    # mock classification
    preds0 = round(random.randint(0, 99) / 100, 2)
    preds1 = round(1 - preds0, 2)
    if preds0 > preds1:
        pred = 0
    else:
        pred = 1

    result = {
        "model_name": "SVM",
        "zero_proba": str(preds0),
        "one_proba": str(preds1),
        "label": str(pred),
    }

    return result


# todo
def predict_nb(tweet):
    # load model

    # clean tweet before vectorization

    # vectorize tweet

    # generate classification
    # probs = model.predict_proba(tweet_vectorized)
    # preds0 = round(probs[:, 0][0], 2)
    # preds1 = round(probs[:, 1][0], 2)
    # pred = model.predict(tweet_vectorized)[0]

    # mock classification
    preds0 = round(random.randint(0, 99) / 100, 2)
    preds1 = round(1 - preds0, 2)
    if preds0 > preds1:
        pred = 0
    else:
        pred = 1

    result = {
        "model_name": "Naive Bayes",
        "zero_proba": str(preds0),
        "one_proba": str(preds1),
        "label": str(pred),
    }

    return result


# todo
def predict_gru(tweet):
    # load model

    # clean tweet before vectorization

    # vectorize tweet

    # generate classification
    # probs = model.predict_proba(tweet_vectorized)
    # preds0 = round(probs[:, 0][0], 2)
    # preds1 = round(probs[:, 1][0], 2)
    # pred = model.predict(tweet_vectorized)[0]

    # mock classification
    preds0 = round(random.randint(0, 99) / 100, 2)
    preds1 = round(1 - preds0, 2)
    if preds0 > preds1:
        pred = 0
    else:
        pred = 1

    result = {
        "model_name": "RNN-GRU",
        "zero_proba": str(preds0),
        "one_proba": str(preds1),
        "label": str(pred),
    }

    return result


# todo
def predict_lstm(tweet):
    # load model

    # clean tweet before vectorization

    # vectorize tweet

    # generate classification
    # probs = model.predict_proba(tweet_vectorized)
    # preds0 = round(probs[:, 0][0], 2)
    # preds1 = round(probs[:, 1][0], 2)
    # pred = model.predict(tweet_vectorized)[0]

    # mock classification
    preds0 = round(random.randint(0, 99) / 100, 2)
    preds1 = round(1 - preds0, 2)
    if preds0 > preds1:
        pred = 0
    else:
        pred = 1

    result = {
        "model_name": "RNN-LSTM",
        "zero_proba": str(preds0),
        "one_proba": str(preds1),
        "label": str(pred),
    }

    return result


# todo
def predict_bert(tweet):
    # load model

    # clean tweet before vectorization

    # vectorize tweet

    # generate classification
    # probs = model.predict_proba(tweet_vectorized)
    # preds0 = round(probs[:, 0][0], 2)
    # preds1 = round(probs[:, 1][0], 2)
    # pred = model.predict(tweet_vectorized)[0]

    # mock classification
    preds0 = round(random.randint(0, 99) / 100, 2)
    preds1 = round(1 - preds0, 2)
    if preds0 > preds1:
        pred = 0
    else:
        pred = 1

    result = {
        "model_name": "BERT",
        "zero_proba": str(preds0),
        "one_proba": str(preds1),
        "label": str(pred),
    }

    return result


# todo
def predict_roberta(tweet):
    # load model

    # clean tweet before vectorization

    # vectorize tweet

    # generate classification
    # probs = model.predict_proba(tweet_vectorized)
    # preds0 = round(probs[:, 0][0], 2)
    # preds1 = round(probs[:, 1][0], 2)
    # pred = model.predict(tweet_vectorized)[0]

    # mock classification
    preds0 = round(random.randint(0, 99) / 100, 2)
    preds1 = round(1 - preds0, 2)
    if preds0 > preds1:
        pred = 0
    else:
        pred = 1

    result = {
        "model_name": "ROBERTA",
        "zero_proba": str(preds0),
        "one_proba": str(preds1),
        "label": str(pred),
    }

    return result
