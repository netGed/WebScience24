import joblib
import random
import pandas as pd
import torch
from scipy.special import softmax
from transformers import AutoTokenizer, AutoModelForSequenceClassification


tfidf_vectorizer_mixeddata = joblib.load("models/ensemble/tfidf_vectorizer_for_brf.joblib")


def classify_with_ensemble(tweet):
    # load model
    model = joblib.load("models/ensemble/brf_untuned_tfidf_model.joblib")

    # clean tweet before vectorization
    # todo?

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
        "zero_probability": str(preds0),
        "one_probability": str(preds1),
        "label": str(pred),
    }

    return result


# todo
def classify_with_svm(tweet):
    # load model

    # clean tweet before vectorization

    # vectorize tweet

    # generate classification
    # probs = model.classify_with_proba(tweet_vectorized)
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
        "zero_probability": str(preds0),
        "one_probability": str(preds1),
        "label": str(pred),
    }

    return result


# todo
def classify_with_nb(tweet):
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
        "zero_probability": str(preds0),
        "one_probability": str(preds1),
        "label": str(pred),
    }

    return result


# todo
def classify_with_gru(tweet):
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
        "zero_probability": str(preds0),
        "one_probability": str(preds1),
        "label": str(pred),
    }

    return result


# todo
def classify_with_lstm(tweet):
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
        "zero_probability": str(preds0),
        "one_probability": str(preds1),
        "label": str(pred),
    }

    return result


# todo
def classify_with_bert(tweet):
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
        "zero_probability": str(preds0),
        "one_probability": str(preds1),
        "label": str(pred),
    }

    return result


ROBERTA_MODEL_PATH = f"models/roberta/cardiffnlp/twitter-roberta-base-sentiment"
tokenizer_roberta = AutoTokenizer.from_pretrained(ROBERTA_MODEL_PATH, map_location=torch.device('cpu'),
                                                  local_files_only=True)
model_roberta = AutoModelForSequenceClassification.from_pretrained(ROBERTA_MODEL_PATH, local_files_only=True)


def classify_with_roberta(tweet):
    # tokenize input
    encoded_input = tokenizer_roberta(tweet, return_tensors='pt')

    # generate classification
    output = model_roberta(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    # generate classification
    # Indizes: ["negative", "neutral", "positive"]
    preds0 = round(scores[2], 2)  # 2 = positive
    preds1 = round(scores[0], 2)  # 0 = negative
    if preds0 > preds1:
        pred = 0
    else:
        pred = 1

    result = {
        "model_name": "ROBERTA",
        "zero_probability": str(preds0),
        "one_probability": str(preds1),
        "label": str(pred),
    }

    return result
