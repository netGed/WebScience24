import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import numpy as np
from gensim.models import FastText
from sklearn.model_selection import train_test_split



def vectorize_bow():

    df = pd.read_csv('../../../data/twitter_hate-speech/train_cleaned.csv', index_col=0)
    df = df[df['tweet_cleaned'].notna()]

    X_base = df.tweet_cleaned
    y_base = df.label

    X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(X_base, y_base, test_size=0.3,
                                                                            random_state=42)

    bow_vectorizer = CountVectorizer()

    X_train_bow = bow_vectorizer.fit_transform(X_train_base)
    X_test_bow = bow_vectorizer.transform(X_test_base)

    return X_train_bow, X_test_bow, y_train_base, y_test_base


def vectorize_tfidf():
    df = pd.read_csv('../../../data/twitter_hate-speech/train_cleaned.csv', index_col=0)
    df = df[df['tweet_cleaned'].notna()]

    X_base = df.tweet_cleaned
    y_base = df.label

    X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(X_base, y_base, test_size=0.3,
                                                                            random_state=42)

    tfidf_vectorizer = TfidfVectorizer()

    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_base)
    X_test_tfidf = tfidf_vectorizer.transform(X_test_base)

    return X_train_tfidf, X_test_tfidf, y_train_base, y_test_base


def vectorize_w2v():
    df = pd.read_csv('../../../data/twitter_hate-speech/train_cleaned.csv', index_col=0)
    df = df[df['tweet_cleaned'].notna()]

    X_base = df.tweet_cleaned
    y_base = df.label

    X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(X_base, y_base, test_size=0.3,
                                                                            random_state=42)

    X_train_base_tokenized = X_train_base.map(word_tokenize)
    X_test_base_tokenized = X_test_base.map(word_tokenize)

    w2v = Word2Vec(X_train_base, window=4, min_count=1, sg=0)
    w2v.train(X_train_base_tokenized, total_examples=len(X_train_base), epochs=20)

    def w2v_vector(tokenized_tweet, size):
        vec = np.zeros(size).reshape((1, size))
        count = 0
        for word in tokenized_tweet:
            try:
                vec += w2v.wv[word].reshape((1, size))
                count += 1
            except KeyError:

                continue
        if count != 0:
            vec /= count
        return vec

    size = 100
    X_train_w2v = np.zeros((len(X_train_base_tokenized), size))
    for i in range(len(X_train_base_tokenized)):
        X_train_w2v[i, :] = w2v_vector(X_train_base_tokenized.iloc[i], size)

    X_test_w2v = np.zeros((len(X_test_base_tokenized), size))
    for i in range(len(X_test_base_tokenized)):
        X_test_w2v[i, :] = w2v_vector(X_test_base_tokenized.iloc[i], size)

    return X_train_w2v, X_test_w2v, y_train_base, y_test_base




def vectorize_ft():
    def ft_vector(tokenized_tweet, size):
        vec = np.zeros(size).reshape((1, size))
        count = 0
        for word in tokenized_tweet:
            try:
                vec += ft.wv[word].reshape((1, size))
                count += 1
            except KeyError:

                continue
        if count != 0:
            vec /= count
        return vec

    df = pd.read_csv('../../../data/twitter_hate-speech/train_cleaned.csv', index_col=0)
    df = df[df['tweet_cleaned'].notna()]

    X_base = df.tweet_cleaned
    y_base = df.label

    X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(X_base, y_base, test_size=0.3,
                                                                            random_state=42)

    X_train_base_tokenized = X_train_base.map(word_tokenize)
    X_test_base_tokenized = X_test_base.map(word_tokenize)

    ft = FastText(X_train_base, window=4)
    ft.train(X_train_base_tokenized, total_examples=len(X_train_base), epochs=20)

    size = 100
    X_train_ft = np.zeros((len(X_train_base_tokenized), size))
    for i in range(len(X_train_base_tokenized)):
        X_train_ft[i, :] = ft_vector(X_train_base_tokenized.iloc[i], size)

    X_test_ft = np.zeros((len(X_test_base_tokenized), size))
    for i in range(len(X_test_base_tokenized)):
        X_test_ft[i, :] = ft_vector(X_test_base_tokenized.iloc[i], size)

    return X_train_ft, X_test_ft, y_train_base, y_test_base
