import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from gensim.models import Word2Vec
import numpy as np
from gensim.models import FastText
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def vectorize_bag_of_words(df, text_column, label_column, test_size=0.3, random_state=42):
    """
    Vectorizes text data using Bag of Words (BoW) and splits it into training and test sets.

    Args:
        df (pd.DataFrame): The DataFrame containing the text and label data.
        text_column (str): The name of the column containing text data to vectorize.
        label_column (str): The name of the column containing the target labels.
        test_size (float, optional): The proportion of the data to use as test data (default is 0.3).
        random_state (int, optional): Random state for reproducibility (default is 42).

    Returns:
        X_train_bow (scipy.sparse.csr_matrix): The BoW vectorized training data.
        X_test_bow (scipy.sparse.csr_matrix): The BoW vectorized test data.
        y_train (pd.Series): The training labels.
        y_test (pd.Series): The test labels.
        vectorizer (CountVectorizer): The fitted CountVectorizer instance.
    """
   
    df = df[df[text_column].notna()]


    X = df[text_column]
    y = df[label_column]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

  
    vectorizer = CountVectorizer()

    X_train_bow = vectorizer.fit_transform(X_train)
    X_test_bow = vectorizer.transform(X_test)

    return X_train_bow, X_test_bow, y_train, y_test, vectorizer


def vectorize_tfidf(df, text_column, label_column, test_size=0.3, random_state=42):
    """
    Vectorizes text data using TF-IDF and splits it into training and test sets.

    Args:
        df (pd.DataFrame): The DataFrame containing the text and label data.
        text_column (str): The name of the column containing text data to vectorize.
        label_column (str): The name of the column containing the target labels.
        test_size (float, optional): The proportion of the data to use as test data (default is 0.3).
        random_state (int, optional): Random state for reproducibility (default is 42).

    Returns:
        X_train_tfidf (scipy.sparse.csr_matrix): The TF-IDF vectorized training data.
        X_test_tfidf (scipy.sparse.csr_matrix): The TF-IDF vectorized test data.
        y_train (pd.Series): The training labels.
        y_test (pd.Series): The test labels.
        tfidf_vectorizer (TfidfVectorizer): The fitted TfidfVectorizer instance.
    """
    X = df[text_column]
    y = df[label_column]


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)


    tfidf_vectorizer = TfidfVectorizer()

  
    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
    X_test_tfidf = tfidf_vectorizer.transform(X_test)

    return X_train_tfidf, X_test_tfidf, y_train, y_test, tfidf_vectorizer


def vectorize_word2vec_with_average(df, text_column, label_column, vector_size=300, window=5, min_count=1, test_size=0.3, random_state=42):
    """
    Vectorizes text data using Word2Vec and splits it into training and test sets.

    Args:
        df (pd.DataFrame): The DataFrame containing the text and label data.
        text_column (str): The name of the column containing text data to vectorize.
        label_column (str): The name of the column containing the target labels.
        vector_size (int, optional): The size of the Word2Vec vectors (default is 300).
        window (int, optional): The window size for context words (default is 5).
        min_count (int, optional): Minimum frequency for words to be included in the vocabulary (default is 1).
        test_size (float, optional): The proportion of the data to use as test data (default is 0.3).
        random_state (int, optional): Random state for reproducibility (default is 42).

    Returns:
        X_train (np.ndarray): Word2Vec vectorized training data.
        X_test (np.ndarray): Word2Vec vectorized test data.
        y_train (pd.Series): The training labels.
        y_test (pd.Series): The test labels.
        w2v_model (gensim.models.Word2Vec): The trained Word2Vec model.
    """
    df = df[df[text_column].notna()]
    df[text_column] = df[text_column].astype(str)


    X = df[text_column]
    y = df[label_column]

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Tokenize the text data
    X_train_tokenized = X_train.map(lambda x: word_tokenize(x) if isinstance(x, str) else [])
    X_test_tokenized = X_test.map(lambda x: word_tokenize(x) if isinstance(x, str) else [])

    w2v_model = Word2Vec(sentences=X_train_tokenized, vector_size=vector_size, window=window, min_count=min_count, sg=0)

 
    def average_word_vectors(tokenized_sentence, model, vector_size):
        vec = np.zeros(vector_size)
        count = 0
        for word in tokenized_sentence:
            if word in model.wv:
                vec += model.wv[word]
                count += 1
        if count > 0:
            vec /= count
        return vec

    X_train_vectors = np.array([average_word_vectors(sentence, w2v_model, vector_size) for sentence in X_train_tokenized])
    X_test_vectors = np.array([average_word_vectors(sentence, w2v_model, vector_size) for sentence in X_test_tokenized])

    return X_train_vectors, X_test_vectors, y_train, y_test, w2v_model

def vectorize_word2vec(df, text_column, label_column, vector_size=100, max_seq_len=50, window=5, min_count=1, test_size=0.3, random_state=42):
    """
    Vectorizes text data using Word2Vec and returns sequences of word vectors.

    Args:
        df (pd.DataFrame): The DataFrame containing the text and label data.
        text_column (str): The name of the column containing text data to vectorize.
        label_column (str): The name of the column containing the target labels.
        vector_size (int, optional): The size of the Word2Vec vectors (default is 300).
        max_seq_len (int, optional): The maximum sequence length for padding/truncation (default is 50).
        window (int, optional): The window size for context words (default is 5).
        min_count (int, optional): Minimum frequency for words to be included in the vocabulary (default is 1).
        test_size (float, optional): The proportion of the data to use as test data (default is 0.3).
        random_state (int, optional): Random state for reproducibility (default is 42).

    Returns:
        X_train (np.ndarray): Word2Vec vectorized training data as sequences.
        X_test (np.ndarray): Word2Vec vectorized test data as sequences.
        y_train (pd.Series): The training labels.
        y_test (pd.Series): The test labels.
        w2v_model (gensim.models.Word2Vec): The trained Word2Vec model.
    """
    # Step 1: Remove missing values and ensure all text is a string
    df = df[df[text_column].notna()]
    df[text_column] = df[text_column].astype(str)

    X = df[text_column]
    y = df[label_column]

    # Step 2: Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Step 3: Tokenize the text data
    X_train_tokenized = X_train.map(lambda x: word_tokenize(x) if isinstance(x, str) else [])
    X_test_tokenized = X_test.map(lambda x: word_tokenize(x) if isinstance(x, str) else [])

    # Step 4: Train Word2Vec model
    w2v_model = Word2Vec(sentences=X_train_tokenized, vector_size=vector_size, window=window, min_count=min_count, sg=0)

    # Step 5: Convert tokenized sentences to sequences of vectors
    def get_word2vec_sequence(tokenized_sentence, model, vector_size, max_seq_len):
        vectors = []
        for word in tokenized_sentence:
            if word in model.wv:
                vectors.append(model.wv[word])
            else:
                vectors.append(np.zeros(vector_size))  # Use zero vector for unknown words
        # Truncate or pad the sequence to the desired max_seq_len
        if len(vectors) > max_seq_len:
            vectors = vectors[:max_seq_len]
        else:
            vectors.extend([np.zeros(vector_size)] * (max_seq_len - len(vectors)))
        return np.array(vectors)

    X_train_vectors = np.array([get_word2vec_sequence(sentence, w2v_model, vector_size, max_seq_len) for sentence in X_train_tokenized])
    X_test_vectors = np.array([get_word2vec_sequence(sentence, w2v_model, vector_size, max_seq_len) for sentence in X_test_tokenized])

    return X_train_vectors, X_test_vectors, y_train, y_test, w2v_model


def vectorize_fasttext(df, text_column, label_column, vector_size=300, window=5, min_count=1, test_size=0.3, random_state=42):
    """
    Vectorize text using FastText.

    Args:
    - df (pd.DataFrame): The DataFrame containing the text data.
    - text_column (str): Name of the column with text data.
    - label_column (str): Name of the column with labels.
    - vector_size (int): Dimensionality of the vector embeddings.
    - window (int): Context window size.
    - min_count (int): Minimum count for a word to be considered.
    - test_size (float): Proportion of the data to be used for the test set.
    - random_state (int): Random seed for splitting the data.

    Returns:
    - X_train_ft (np.ndarray): FastText vectors for the training set.
    - X_test_ft (np.ndarray): FastText vectors for the test set.
    - y_train (pd.Series): Labels for the training set.
    - y_test (pd.Series): Labels for the test set.
    - ft_model (FastText): The trained FastText model.
    """
    
   
    df = df[df[text_column].notna()]
    
   
    X = df[text_column]
    y = df[label_column]
    
   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
   
    X_train_tokenized = X_train.map(word_tokenize)
    X_test_tokenized = X_test.map(word_tokenize)
    
   
    ft_model = FastText(sentences=X_train_tokenized, vector_size=vector_size, window=window, min_count=min_count, sg=1, epochs=10)
    
   
    def fasttext_vector(tokenized_sentence, model, vector_size):
        vec = np.zeros(vector_size).reshape((1, vector_size))
        count = 0
        for word in tokenized_sentence:
            try:
                vec += model.wv[word].reshape((1, vector_size))
                count += 1
            except KeyError:
                continue
        if count != 0:
            vec /= count
        return vec

    X_train_ft = np.zeros((len(X_train_tokenized), vector_size))
    for i, sentence in enumerate(X_train_tokenized):
        X_train_ft[i, :] = fasttext_vector(sentence, ft_model, vector_size)
    
    
    X_test_ft = np.zeros((len(X_test_tokenized), vector_size))
    for i, sentence in enumerate(X_test_tokenized):
        X_test_ft[i, :] = fasttext_vector(sentence, ft_model, vector_size)
    
    return X_train_ft, X_test_ft, y_train, y_test, ft_model

def vectorize_glove_with_average(df, text_column, label_column, glove_path, vector_size=100, test_size=0.3, random_state=42):
    """
    Vectorizes text data using pre-trained GloVe embeddings and splits it into training and test sets.

    Args:
        df (pd.DataFrame): The DataFrame containing the text and label data.
        text_column (str): The name of the column containing text data to vectorize.
        label_column (str): The name of the column containing the target labels.
        glove_path (str): Path to the pre-trained GloVe embeddings file.
        vector_size (int, optional): The size of the GloVe vectors (default is 200 for glove.twitter.27B.200d.txt).
        test_size (float, optional): The proportion of the data to use as test data (default is 0.3).
        random_state (int, optional): Random state for reproducibility (default is 42).

    Returns:
        X_train (np.ndarray): GloVe vectorized training data.
        X_test (np.ndarray): GloVe vectorized test data.
        y_train (pd.Series): The training labels.
        y_test (pd.Series): The test labels.
        glove_embeddings (dict): The loaded GloVe embeddings dictionary.
    """
    # Step 1: Remove missing values and ensure all text is a string
    df = df[df[text_column].notna()]
    df[text_column] = df[text_column].astype(str)

    # Step 2: Load GloVe embeddings into a dictionary
    glove_embeddings = {}
    with open(glove_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            glove_embeddings[word] = vector

    # Step 3: Split the data into training and test sets
    X = df[text_column]
    y = df[label_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Step 4: Tokenize the text data
    X_train_tokenized = X_train.map(word_tokenize)
    X_test_tokenized = X_test.map(word_tokenize)

    # Step 5: Function to calculate the average GloVe vector for a tokenized tweet
    def average_glove_vector(tokenized_tweet, glove_embeddings, vector_size):
        vec = np.zeros(vector_size)
        count = 0
        for word in tokenized_tweet:
            if word in glove_embeddings:
                vec += glove_embeddings[word]
                count += 1
        if count > 0:
            vec /= count
        return vec

    # Step 6: Convert tokenized tweets to vectors
    X_train_vectors = np.array([average_glove_vector(tweet, glove_embeddings, vector_size) for tweet in X_train_tokenized])
    X_test_vectors = np.array([average_glove_vector(tweet, glove_embeddings, vector_size) for tweet in X_test_tokenized])

    return X_train_vectors, X_test_vectors, y_train, y_test, glove_embeddings 

def vectorize_glove(df, text_column, label_column, glove_path, vector_size=100, max_seq_len=50, test_size=0.3, random_state=42):
    """
    Vectorizes text data using pre-trained GloVe embeddings and splits it into training and test sets.
    Each sentence is represented as a sequence of word vectors.

    Args:
        df (pd.DataFrame): The DataFrame containing the text and label data.
        text_column (str): The name of the column containing text data to vectorize.
        label_column (str): The name of the column containing the target labels.
        glove_path (str): Path to the pre-trained GloVe embeddings file.
        vector_size (int, optional): The size of the GloVe vectors (default is 100).
        max_seq_len (int, optional): The maximum sequence length for padding (default is 50).
        test_size (float, optional): The proportion of the data to use as test data (default is 0.3).
        random_state (int, optional): Random state for reproducibility (default is 42).

    Returns:
        X_train (np.ndarray): GloVe vectorized training data.
        X_test (np.ndarray): GloVe vectorized test data.
        y_train (pd.Series): The training labels.
        y_test (pd.Series): The test labels.
        glove_embeddings (dict): The loaded GloVe embeddings dictionary.
    """
    # Step 1: Remove missing values and ensure all text is a string
    df = df[df[text_column].notna()]
    df[text_column] = df[text_column].astype(str)

    # Step 2: Load GloVe embeddings into a dictionary
    glove_embeddings = {}
    with open(glove_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            glove_embeddings[word] = vector

    # Step 3: Split the data into training and test sets
    X = df[text_column]
    y = df[label_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Step 4: Tokenize the text data
    X_train_tokenized = X_train.map(word_tokenize)
    X_test_tokenized = X_test.map(word_tokenize)

    # Step 5: Function to convert a tokenized sentence into a sequence of GloVe vectors
    def get_glove_vectors(tokenized_sentence, glove_embeddings, vector_size, max_seq_len):
        vectors = []
        for word in tokenized_sentence:
            if word in glove_embeddings:
                vectors.append(glove_embeddings[word])
            else:
                vectors.append(np.zeros(vector_size))  # Use zero vector for unknown words
        # Truncate or pad the sequence to the desired max_seq_len
        if len(vectors) > max_seq_len:
            vectors = vectors[:max_seq_len]
        else:
            vectors.extend([np.zeros(vector_size)] * (max_seq_len - len(vectors)))
        return np.array(vectors)

    # Step 6: Convert tokenized sentences to sequences of vectors
    X_train_vectors = np.array([get_glove_vectors(tweet, glove_embeddings, vector_size, max_seq_len) for tweet in X_train_tokenized])
    X_test_vectors = np.array([get_glove_vectors(tweet, glove_embeddings, vector_size, max_seq_len) for tweet in X_test_tokenized])

    return X_train_vectors, X_test_vectors, y_train, y_test, glove_embeddings


def compare_vectorization_methods(df, text_column, label_column):
    """
    Compares vectorizing methods BoW, TF-IDF, Word2Vec und FastText.
    """
    results = []

    # 1. Bag of Words
    print("Vektorisierung: Bag of Words")
    X_train_bow, X_test_bow, y_train, y_test, _ = vectorize_bag_of_words(df, text_column, label_column)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train_bow, y_train)
    y_pred_bow = clf.predict(X_test_bow)
    results.append({
        "Method": "Bag of Words",
        "Accuracy": accuracy_score(y_test, y_pred_bow),
        "Precision": precision_score(y_test, y_pred_bow, average="weighted"),
        "Recall": recall_score(y_test, y_pred_bow, average="weighted"),
        "F1 Score": f1_score(y_test, y_pred_bow, average="weighted")
    })

    # 2. TF-IDF
    print("Vektorisierung: TF-IDF")
    X_train_tfidf, X_test_tfidf, y_train, y_test, _ = vectorize_tfidf(df, text_column, label_column)
    clf.fit(X_train_tfidf, y_train)
    y_pred_tfidf = clf.predict(X_test_tfidf)
    results.append({
        "Method": "TF-IDF",
        "Accuracy": accuracy_score(y_test, y_pred_tfidf),
        "Precision": precision_score(y_test, y_pred_tfidf, average="weighted"),
        "Recall": recall_score(y_test, y_pred_tfidf, average="weighted"),
        "F1 Score": f1_score(y_test, y_pred_tfidf, average="weighted")
    })

    # 3. Word2Vec
    print("Vektorisierung: Word2Vec")
    X_train_w2v, X_test_w2v, y_train, y_test, _ = vectorize_word2vec(df, text_column, label_column)
    clf.fit(X_train_w2v, y_train)
    y_pred_w2v = clf.predict(X_test_w2v)
    results.append({
        "Method": "Word2Vec",
        "Accuracy": accuracy_score(y_test, y_pred_w2v),
        "Precision": precision_score(y_test, y_pred_w2v, average="weighted"),
        "Recall": recall_score(y_test, y_pred_w2v, average="weighted"),
        "F1 Score": f1_score(y_test, y_pred_w2v, average="weighted")
    })

    # 4. FastText
    print("Vektorisierung: FastText")
    X_train_ft, X_test_ft, y_train, y_test, _ = vectorize_fasttext(df, text_column, label_column)
    clf.fit(X_train_ft, y_train)
    y_pred_ft = clf.predict(X_test_ft)
    results.append({
        "Method": "FastText",
        "Accuracy": accuracy_score(y_test, y_pred_ft),
        "Precision": precision_score(y_test, y_pred_ft, average="weighted"),
        "Recall": recall_score(y_test, y_pred_ft, average="weighted"),
        "F1 Score": f1_score(y_test, y_pred_ft, average="weighted")
    })

    results_df = pd.DataFrame(results)
    return results_df