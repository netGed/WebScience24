from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


def get_tfidf_vectorized(x_train, x_test):
    vectorizer = TfidfVectorizer(max_features=5000)
    x_train_vectorized = vectorizer.fit_transform(x_train)
    x_test_vectorized = vectorizer.transform(x_test)
    return x_train_vectorized, x_test_vectorized


def get_count_vectorized(x_train, x_test):
    vectorizer = CountVectorizer(max_features=5000)
    x_train_vectorized = vectorizer.fit_transform(x_train)
    x_test_vectorized = vectorizer.transform(x_test)
    return x_train_vectorized, x_test_vectorized
