import random
import pandas as pd
from keras.src.utils import pad_sequences
from scipy.special import softmax

from webapp.backend.app.models.models import tokenizer_roberta, model_roberta, tokenizer_bert, model_bert, \
    model_ensemble, tfidf_vectorizer_ensemble, tokenizer_gru, model_gru, max_len_gru, \
    vectorize_w2v, w2v_vectorizer_nb, model_nb_w2v


def classify_with_ensemble(tweet):
    # load model
    model = model_ensemble

    # vectorize tweet
    df = pd.DataFrame({'Tweet': [tweet]})
    tweet_vectorized = tfidf_vectorizer_ensemble.transform([df['Tweet'][0]])

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


def classify_with_nb(tweet):
    # vectorize tweet
    df = pd.DataFrame([tweet])
    tweet_vectorized = vectorize_w2v(df[0], w2v_vectorizer_nb)

    # generate classification
    probs = model_nb_w2v.predict_proba(tweet_vectorized)
    preds0 = round(probs[:, 0][0], 2)
    preds1 = round(probs[:, 1][0], 2)
    pred = model_nb_w2v.predict(tweet_vectorized)[0]

    result = {
        "model_name": "Naive Bayes",
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
        "zero_probability": str(preds0),
        "one_probability": str(preds1),
        "label": str(pred),
    }

    return result


def classify_with_gru(tweet):
    # generate classification
    tweet_seq = tokenizer_gru.texts_to_sequences([tweet])
    tweet_padded = pad_sequences(tweet_seq, padding='post', maxlen=max_len_gru)
    pred = model_gru.predict(tweet_padded)[0][0]

    # mock classification
    preds0 = round(1 - pred, 2)
    preds1 = round(pred, 2)
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


def classify_with_bert(tweet):
    # tokenize input
    encoded_input = tokenizer_bert(tweet, return_tensors='pt')

    # generate classification
    output = model_bert(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    # generate classification
    preds0 = round(scores[0], 2)
    preds1 = round(scores[1], 2)
    if preds0 > 0.5:
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


def classify_with_roberta(tweet):
    # tokenize input
    encoded_input = tokenizer_roberta(tweet, return_tensors='pt')

    # generate classification
    output = model_roberta(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    # generate classification
    preds0 = round(scores[0], 2)
    preds1 = round(scores[1], 2)
    if preds0 > 0.5:
        pred = 0
    else:
        pred = 1

    result = {
        "model_name": "RoBERTa",
        "zero_probability": str(preds0),
        "one_probability": str(preds1),
        "label": str(pred),
    }

    return result
