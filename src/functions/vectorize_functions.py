import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from gensim.models import Word2Vec
import numpy as np
from sklearn.model_selection import train_test_split
import gensim.downloader as api

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
        y_train_tfidf (pd.Series): The training labels.
        y_test_tfidf (pd.Series): The test labels.
    """
    df = df[df[text_column].notna()]

    X_base = df[text_column]
    y_base = df[label_column]

    X_train_base, X_test_base, y_train_tfidf, y_test_tfidf = train_test_split(X_base, y_base, test_size=test_size,
                                                                              random_state=random_state)

    tfidf_vectorizer = TfidfVectorizer()

    X_train_tfidf = tfidf_vectorizer.fit_transform(X_train_base)
    X_test_tfidf = tfidf_vectorizer.transform(X_test_base)

    return X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf, tfidf_vectorizer
    
    
def vectorize_w2v(df, text_column, label_column, vector_size=300, window=5, min_count=1, test_size=0.3, random_state=42):
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
        X_train_w2v (np.ndarray): Word2Vec vectorized training data.
        X_test_w2v (np.ndarray): Word2Vec vectorized test data.
        y_train_w2v (pd.Series): The training labels.
        y_test_w2v (pd.Series): The test labels.
    """

    df = df[df[text_column].notna()]
    df.loc[:,text_column] = df[text_column].astype(str)

    X_base = df[text_column]
    y_base = df[label_column]

    X_train_base, X_test_base, y_train_w2v, y_test_w2v = train_test_split(X_base, y_base, test_size=test_size, random_state=random_state)

    X_train_base_tokenized = X_train_base.map(word_tokenize)
    X_test_base_tokenized = X_test_base.map(word_tokenize)

    w2v = Word2Vec(min_count=min_count, window=window, vector_size=vector_size, sg=0)
    w2v.build_vocab(X_train_base_tokenized)#, progress_per=10000)
    w2v.train(X_train_base_tokenized, total_examples=len(X_train_base_tokenized), epochs=30)

    def w2v_vector(tokenized_tweet, vector_size):
        vec = np.zeros(vector_size).reshape((1, vector_size))
        count = 0
        for word in tokenized_tweet:
            try:
                vec += w2v.wv[word].reshape((1, vector_size))
                count += 1
            except KeyError:

                continue
        if count != 0:
            vec /= count
        return vec

    X_train_w2v = np.zeros((len(X_train_base_tokenized), vector_size))
    for i in range(len(X_train_base_tokenized)):
        X_train_w2v[i, :] = w2v_vector(X_train_base_tokenized.iloc[i], vector_size)

    X_test_w2v = np.zeros((len(X_test_base_tokenized), vector_size))
    for i in range(len(X_test_base_tokenized)):
        X_test_w2v[i, :] = w2v_vector(X_test_base_tokenized.iloc[i], vector_size)

    return X_train_w2v, X_test_w2v, y_train_w2v, y_test_w2v, w2v
    

def vectorize_glove(df, text_column, label_column, vector_size=100, test_size=0.3, random_state=42):
    """
    Vectorizes text data using pre-trained GloVe embeddings and splits it into training and test sets.

    Args:
        df (pd.DataFrame): The DataFrame containing the text and label data.
        text_column (str): The name of the column containing text data to vectorize.
        label_column (str): The name of the column containing the target labels.
        vector_size (int, optional): The size of the GloVe vectors (default is 200 for glove.twitter.27B.200d.txt).
        test_size (float, optional): The proportion of the data to use as test data (default is 0.3).
        random_state (int, optional): Random state for reproducibility (default is 42).

    Returns:
        X_train_glv (np.ndarray): GloVe vectorized training data.
        X_test_glv (np.ndarray): GloVe vectorized test data.
        y_train_glv (pd.Series): The training labels.
        y_test_glv (pd.Series): The test labels.
    """
    # Step 1: Remove missing values and ensure all text is a string
    df = df[df[text_column].notna()]
    df.loc[:,text_column] = df[text_column].astype(str)

    # Step 2: Load GloVe model
    glv_model = api.load("glove-twitter-100")

    # Step 3: Split the data into training and test sets
    X_base = df[text_column]
    y_base = df[label_column]
    X_train, X_test, y_train_glv, y_test_glv = train_test_split(X_base, y_base, test_size=test_size, random_state=random_state)

    # Step 4: Tokenize the text data
    X_train_tokenized = X_train.map(word_tokenize)
    X_test_tokenized = X_test.map(word_tokenize)

    # Step 5: Function to calculate the average GloVe vector for a tokenized tweet
    def average_glove_vector(tokenized_tweet, glv_model, vector_size):
        vec = np.zeros(vector_size)
        count = 0
        for word in tokenized_tweet:
            if word in glv_model:
                vec += glv_model[word]
                count += 1
        if count > 0:
            vec /= count
        return vec

    # Step 6: Convert tokenized tweets to vectors
    X_train_glv = np.array([average_glove_vector(tweet, glv_model, vector_size) for tweet in X_train_tokenized])
    X_test_glv = np.array([average_glove_vector(tweet, glv_model, vector_size) for tweet in X_test_tokenized])

    return X_train_glv, X_test_glv, y_train_glv, y_test_glv, glv_model

def vectorize_glove_without_avarage(df, text_column, label_column, glove_path, vector_size=100, max_seq_len=50, test_size=0.3, random_state=42):
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

def vectorize_word2vec_no_average(df, text_column, label_column, vector_size=100, max_seq_len=50, window=5, min_count=1, test_size=0.3, random_state=42):
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

def vectorize_glove_test_data(df, text_column, label_column, glove_path, vector_size=200, max_seq_len=50):
    """
    Vectorizes text data using pre-trained GloVe embeddings.
    Each sentence is represented as a sequence of word vectors.

    Args:
        df (pd.DataFrame): The DataFrame containing the text and labels.
        text_column (str): The name of the column containing text data to vectorize.
        label_column (str): The name of the column containing target labels.
        glove_path (str): Path to the pre-trained GloVe embeddings file.
        vector_size (int, optional): The size of the GloVe vectors (default is 100).
        max_seq_len (int, optional): The maximum sequence length for padding (default is 50).

    Returns:
        train_vectors (np.ndarray): GloVe vektorisierte Daten.
        train_y (pd.Series): Labels der Trainingsdaten.
        glove_embeddings (dict): Das geladene GloVe-Embedding-Wörterbuch.
    """
    # Step 1: Entferne fehlende Werte und stelle sicher, dass der Text als String vorliegt
    df = df[df[text_column].notna()]
    df[text_column] = df[text_column].astype(str)

    # Step 2: Lade GloVe-Embeddings in ein Wörterbuch
    glove_embeddings = {}
    with open(glove_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            glove_embeddings[word] = vector

    # Step 3: Tokenisiere den Text
    X_tokenized = df[text_column].map(word_tokenize)

    # Step 4: Funktion zur Umwandlung eines tokenisierten Satzes in eine Sequenz von GloVe-Vektoren
    def get_glove_vectors(tokenized_sentence, glove_embeddings, vector_size, max_seq_len):
        vectors = []
        for word in tokenized_sentence:
            if word in glove_embeddings:
                vectors.append(glove_embeddings[word])
            else:
                vectors.append(np.zeros(vector_size))  # Null-Vektor für unbekannte Wörter
        # Kürze oder padde die Sequenz auf die gewünschte max_seq_len
        if len(vectors) > max_seq_len:
            vectors = vectors[:max_seq_len]
        else:
            vectors.extend([np.zeros(vector_size)] * (max_seq_len - len(vectors)))
        return np.array(vectors)

    # Step 5: Konvertiere tokenisierte Sätze in Vektoren
    train_vectors = np.array([get_glove_vectors(sentence, glove_embeddings, vector_size, max_seq_len) for sentence in X_tokenized])

    # Step 6: Extrahiere die Labels
    train_y = df[label_column]

    return train_vectors, train_y, glove_embeddings