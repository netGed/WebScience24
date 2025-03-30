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
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from gensim.models import FastText

def vectorize_word2vec(df, text_column, label_column, vector_size=200, window=5, min_count=1, test_size=0.3, random_state=42):
    """
    Vectorizes text data using Word2Vec and returns sequences of word vectors.

    Args:
        df (pd.DataFrame): The DataFrame containing the text and label data.
        text_column (str): The name of the column containing text data to vectorize.
        label_column (str): The name of the column containing the target labels.
        vector_size (int, optional): The size of the Word2Vec vectors (default is 200).
        window (int, optional): The window size for context words (default is 5).
        min_count (int, optional): Minimum frequency for words to be included in the vocabulary (default is 1).
        test_size (float, optional): The proportion of the data to use as test data (default is 0.3).
        random_state (int, optional): Random state for reproducibility (default is 42).

    Returns:
        X_train (np.ndarray): Word2Vec vectorized training data as sequences (padded).
        X_test (np.ndarray): Word2Vec vectorized test data as sequences (padded).
        y_train_resampled (pd.Series): The resampled training labels.
        y_test (pd.Series): The test labels.
        w2v_model (gensim.models.Word2Vec): The trained Word2Vec model.
    """
    
    df = df[df[text_column].notna()]
    df[text_column] = df[text_column].astype(str)

    # Datenaufteilung in Training- und Testsets
    X = df[text_column]
    y = df[label_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Oversampling zur Behandlung von Klassenungleichgewicht
    ros = RandomOverSampler(random_state=random_state)

    X_train_resampled, y_train_resampled = ros.fit_resample(X_train.values.reshape(-1, 1), y_train)
    X_train_resampled = X_train_resampled[:, 0]  # Konvertiere zurück zu einer 1D-Array

    print("Shape nach Oversampling:", X_train_resampled.shape)
    print(X_train_resampled.shape)
    print(y_train_resampled.shape)


    # Word2Vec Modell trainieren auf den originalen, nicht gepaddeten Token
    X_train_tokenized = [text.split() for text in X_train_resampled]
    print("Originaltext:", df[text_column].iloc[0])
    print("Tokenisierte Version:", X_train_tokenized[0])  # Liste von Wörtern?
    w2v_model = Word2Vec(sentences=X_train_tokenized, vector_size=vector_size, window=window, min_count=min_count, sg=0,epochs=20)
    
    missing_tokens = []
    for text in X[:100]:  # Nur erste 100 Sätze prüfen
        for word in text.split():
            if word not in w2v_model.wv:
                missing_tokens.append(word)

    print(f"❌ Fehlende Token (Beispiele): {missing_tokens[:20]}")
    print(f"Gesamtanzahl fehlender Token: {len(missing_tokens)}")
    
    # Funktion zur Umwandlung eines Satzes in eine Sequenz von Word2Vec-Vektoren
    def word2vec_vector(text, model, vector_size):
        vectors = []
        for word in text.split():
            if word in model.wv:
                vectors.append(model.wv[word])
            else:
                vectors.append(np.random.uniform(-0.1, 0.1, vector_size))  # Zufällige kleine Werte statt 0-Vektoren
        return np.array(vectors, dtype=np.float32)

    # Konvertiere Texte in Sequenzen von Vektoren
    X_train_w2v = [word2vec_vector(text, w2v_model, vector_size) for text in X_train_resampled]
    print("Länge von X_train_w2v:", len(X_train_w2v))
    print("Erste Sequenz:", X_train_w2v[0] if len(X_train_w2v) > 0 else "LEER")
    X_test_w2v = [word2vec_vector(text, w2v_model, vector_size) for text in X_test]

    # Padding: Bestimme die maximale Sequenzlänge
    max_seq_len = max(len(seq) for seq in X_train_w2v)

    # Padding-Funktion anwenden
    def pad_sequence(sequences, max_len, vector_size):
        return np.array([
            np.vstack([seq, np.zeros((max_len - len(seq), vector_size))]) if len(seq) < max_len else seq[:max_len]
            for seq in sequences
        ])

    # Padding auf Trainings- und Testdaten anwenden
    X_train_w2v_padded = pad_sequence(X_train_w2v, max_seq_len, vector_size)
    X_test_w2v_padded = pad_sequence(X_test_w2v, max_seq_len, vector_size)

    return X_train_w2v_padded, X_test_w2v_padded, y_train_resampled, y_test, w2v_model

def vectorize_word2vec_test_data(df, text_column, label_column, vector_size=200, window=5, min_count=1, w2v_model=None):
    """
    Vectorizes text data using Word2Vec and returns sequences of word vectors.

    Args:
        df (pd.DataFrame): The DataFrame containing the text and labels.
        text_column (str): The name of the column containing text data to vectorize.
        label_column (str): The name of the column containing the target labels.
        vector_size (int, optional): The size of the Word2Vec vectors (default is 200).
        window (int, optional): The window size for context words (default is 5).
        min_count (int, optional): Minimum frequency for words to be included in the vocabulary (default is 1).
        w2v_model (Word2Vec, optional): The trained Word2Vec model. If None, a new one is trained.

    Returns:
        X_w2v (np.ndarray): Word2Vec vectorized data as sequences (padded).
        y_labels (pd.Series): Labels of the data.
        w2v_model (gensim.models.Word2Vec): The trained Word2Vec model.
    """
    # Entferne fehlende Werte und stelle sicher, dass der Text als String vorliegt
    df = df[df[text_column].notna()]
    df[text_column] = df[text_column].astype(str)

    X = df[text_column].tolist()
    y_labels = df[label_column]
    
    # Falls kein trainiertes Modell übergeben wurde, ein neues Modell trainieren
    if w2v_model is None:
        X_tokenized = [text.split() for text in X]
        print("Originaltext:", df[text_column].iloc[0])
        print("Tokenisierte Version:", X[0])  # Liste von Wörtern?

        w2v_model = Word2Vec(sentences=X_tokenized, vector_size=vector_size, window=window, min_count=min_count, sg=0)
    
    missing_tokens = []
    for text in X[:100]:  # Nur erste 100 Sätze prüfen
        for word in text.split():
            if word not in w2v_model.wv:
                missing_tokens.append(word)

    print(f"❌ Fehlende Token (Beispiele): {missing_tokens[:20]}")
    print(f"Gesamtanzahl fehlender Token: {len(missing_tokens)}")
    
    # Funktion zur Umwandlung einer Sequenz in Word2Vec-Vektoren
    def word2vec_vector(text, model, vector_size):
        vectors = []
        for word in text.split():
            if word in model.wv:
                vectors.append(model.wv[word])
            else:
                vectors.append(np.random.uniform(-0.1, 0.1, vector_size))  # Zufällige kleine Werte statt 0-Vektoren
        return np.array(vectors, dtype=np.float32)

    # Wandle Texte in Sequenzen von Word2Vec-Vektoren um
    X_w2v = [word2vec_vector(text, w2v_model, vector_size) for text in X]

    # Padding: Bestimme die maximale Sequenzlänge
    max_seq_len = max(len(seq) for seq in X_w2v)

    # Padding-Funktion anwenden
    def pad_sequence(sequences, max_len, vector_size):
        return np.array([
            np.vstack([seq, np.zeros((max_len - len(seq), vector_size))]) if len(seq) < max_len else seq[:max_len]
            for seq in sequences
        ])

    # Padding auf Testdaten anwenden
    X_w2v_padded = pad_sequence(X_w2v, max_seq_len, vector_size)

    return X_w2v_padded, y_labels, w2v_model



def vectorize_fasttext(df, text_column, label_column, vector_size=200, window=5, min_count=1, test_size=0.3, random_state=42, tokenizer=None):
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
    - tokenizer (Tokenizer, optional): The fitted tokenizer from training. If None, a new tokenizer is created.

    Returns:
    - X_train_ft (np.ndarray): FastText vectors for the training set.
    - X_test_ft (np.ndarray): FastText vectors for the test set.
    - y_train (pd.Series): Labels for the training set.
    - y_test (pd.Series): Labels for the test set.
    - ft_model (FastText): The trained FastText model.
    - tokenizer (Tokenizer): The fitted tokenizer.
    """
    # Entferne fehlende Werte und stelle sicher, dass der Text als String vorliegt
    df = df[df[text_column].notna()]
    df[text_column] = df[text_column].astype(str)
    
    # Daten aufteilen
    X = df[text_column]
    y = df[label_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Tokenizer verwenden oder neuen erstellen
    if tokenizer is None:
        tokenizer = Tokenizer(oov_token="<OOV>")
        tokenizer.fit_on_texts(X_train)
    
    # Tokenisierung der Texte
    X_train_sequences = tokenizer.texts_to_sequences(X_train)
    X_test_sequences = tokenizer.texts_to_sequences(X_test)
    
    # Padding der Sequenzen
    max_seq_len = max(len(seq) for seq in X_train_sequences)
    X_train_padded = pad_sequences(X_train_sequences, padding='post', maxlen=max_seq_len)
    X_test_padded = pad_sequences(X_test_sequences, padding='post', maxlen=max_seq_len)
    
    # FastText Modell trainieren
    X_train_tokenized = [tokenizer.sequences_to_texts([seq])[0].split() for seq in X_train_padded]
    ft_model = FastText(sentences=X_train_tokenized, vector_size=vector_size, window=window, min_count=min_count, sg=1, epochs=10)
    
    # Funktion zur Umwandlung einer Sequenz in FastText-Vektoren
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

    # Umwandlung von Sequenzen in FastText-Vektoren
    X_train_ft = np.array([fasttext_vector(sentence, ft_model, vector_size) for sentence in X_train_tokenized]).reshape(len(X_train_tokenized), vector_size)
    X_test_ft = np.array([fasttext_vector(sentence, ft_model, vector_size) for sentence in X_test_sequences]).reshape(len(X_test_sequences), vector_size)
    
    return X_train_ft, X_test_ft, y_train, y_test, ft_model, tokenizer

def vectorize_fasttext_test_data(df, text_column, label_column, vector_size=200, window=5, min_count=1, tokenizer=None):
    """
    Vectorize text using FastText without train-test split.

    Args:
    - df (pd.DataFrame): The DataFrame containing the text data.
    - text_column (str): Name of the column with text data.
    - label_column (str): Name of the column with labels.
    - vector_size (int): Dimensionality of the vector embeddings.
    - window (int): Context window size.
    - min_count (int): Minimum count for a word to be considered.
    - tokenizer (Tokenizer, optional): The fitted tokenizer from training. If None, a new tokenizer is created.

    Returns:
    - X_ft (np.ndarray): FastText vectors for the dataset.
    - y (pd.Series): Labels for the dataset.
    - ft_model (FastText): The trained FastText model.
    - tokenizer (Tokenizer): The fitted tokenizer.
    """
    # Entferne fehlende Werte und stelle sicher, dass der Text als String vorliegt
    df = df[df[text_column].notna()]
    df[text_column] = df[text_column].astype(str)
    
    # Tokenizer verwenden oder neuen erstellen
    if tokenizer is None:
        tokenizer = Tokenizer(oov_token="<OOV>")
        tokenizer.fit_on_texts(df[text_column])
    
    # Tokenisierung der Texte
    X_sequences = tokenizer.texts_to_sequences(df[text_column])
    
    # Padding der Sequenzen
    max_seq_len = max(len(seq) for seq in X_sequences)
    X_padded = pad_sequences(X_sequences, padding='post', maxlen=max_seq_len)
    
    # FastText Modell trainieren
    X_tokenized = [tokenizer.sequences_to_texts([seq])[0].split() for seq in X_padded]
    ft_model = FastText(sentences=X_tokenized, vector_size=vector_size, window=window, min_count=min_count, sg=1, epochs=10)
    
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
    # Umwandlung von Sequenzen in FastText-Vektoren
    X_ft = np.array([fasttext_vector(sentence, ft_model, vector_size) for sentence in X_tokenized]).reshape(len(X_tokenized), vector_size)
    
    return X_ft, df[label_column], ft_model, tokenizer

def vectorize_fasttext_words(df, text_column, label_column, vector_size=200, window=5, min_count=1, test_size=0.3, random_state=42):
    """
    Vectorizes text using FastText.

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

    # Entferne fehlende Werte und stelle sicher, dass der Text als String vorliegt
    df = df[df[text_column].notna()]
    df[text_column] = df[text_column].astype(str)
    
    # Daten aufteilen
    X = df[text_column]
    y = df[label_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Tokenisierung: Wörter direkt per split() nutzen
    X_train_tokenized = [text.split() for text in X_train]
    X_test_tokenized = [text.split() for text in X_test]

    # FastText Modell trainieren
    ft_model = FastText(sentences=X_train_tokenized, vector_size=vector_size, window=window, min_count=min_count, sg=1, epochs=10)

    # Funktion zur Umwandlung einer Sequenz in FastText-Vektoren
    def fasttext_vector(tokenized_sentence, model, vector_size):
        vec = np.zeros(vector_size)
        count = 0
        for word in tokenized_sentence:
            if word in model.wv:
                vec += model.wv[word]
                count += 1
        if count != 0:
            vec /= count
        return vec

    # Umwandlung von Sequenzen in FastText-Vektoren
    X_train_ft = np.array([fasttext_vector(sentence, ft_model, vector_size) for sentence in X_train_tokenized])
    X_test_ft = np.array([fasttext_vector(sentence, ft_model, vector_size) for sentence in X_test_tokenized])

    return X_train_ft, X_test_ft, y_train, y_test, ft_model

def vectorize_fasttext_test_data_words(df, text_column, label_column, vector_size=200, window=5, min_count=1, ft_model=None):
    """
    Vectorizes text using an existing FastText model without train-test split.

    Args:
    - df (pd.DataFrame): The DataFrame containing the text data.
    - text_column (str): Name of the column with text data.
    - label_column (str): Name of the column with labels.
    - vector_size (int): Dimensionality of the vector embeddings.
    - window (int): Context window size.
    - min_count (int): Minimum count for a word to be considered.
    - ft_model (FastText, optional): The trained FastText model. If None, a new one is trained.

    Returns:
    - X_ft (np.ndarray): FastText vectors for the dataset.
    - y (pd.Series): Labels for the dataset.
    - ft_model (FastText): The trained FastText model.
    """

    # Entferne fehlende Werte und stelle sicher, dass der Text als String vorliegt
    df = df[df[text_column].notna()]
    df[text_column] = df[text_column].astype(str)

    X = df[text_column]
    y_labels = df[label_column]
    
    # Tokenisierung: Wörter direkt per split() nutzen
    X_tokenized = [text.split() for text in X]

    # Falls kein trainiertes Modell übergeben wurde, neues FastText Modell trainieren
    if ft_model is None:
        ft_model = FastText(sentences=X_tokenized, vector_size=vector_size, window=window, min_count=min_count, sg=1, epochs=10)

    # Funktion zur Umwandlung einer Sequenz in FastText-Vektoren
    def fasttext_vector(tokenized_sentence, model, vector_size):
        vec = np.zeros(vector_size)
        count = 0
        for word in tokenized_sentence:
            if word in model.wv:
                vec += model.wv[word]
                count += 1
        if count != 0:
            vec /= count
        return vec

    # Umwandlung von Sequenzen in FastText-Vektoren
    X_ft = np.array([fasttext_vector(sentence, ft_model, vector_size) for sentence in X_tokenized])

    return X_ft, y_labels, ft_model


def vectorize_glove(df, text_column, label_column, glove_path, vector_size=200, max_seq_len=50, test_size=0.3, random_state=42):
    """
    Vectorizes text data using pre-trained GloVe embeddings and splits it into training and test sets.
    Each sentence is represented as a sequence of word vectors.

    Args:
        df (pd.DataFrame): The DataFrame containing the text and label data.
        text_column (str): The name of the column containing text data to vectorize.
        label_column (str): The name of the column containing the target labels.
        glove_path (str): Path to the pre-trained GloVe embeddings file.
        vector_size (int, optional): The size of the GloVe vectors (default is 200).
        max_seq_len (int, optional): The maximum sequence length for padding (default is 50).
        test_size (float, optional): The proportion of the data to use as test data (default is 0.3).
        random_state (int, optional): Random state for reproducibility (default is 42).

    Returns:
        X_train (np.ndarray): GloVe vectorized training data.
        X_test (np.ndarray): GloVe vectorized test data.
        y_train (pd.Series): The training labels.
        y_test (pd.Series): The test labels.
        glove_embeddings (dict): The loaded GloVe embeddings dictionary.
        tokenizer (Tokenizer): The fitted tokenizer object.
    """
    # Entferne fehlende Werte und stelle sicher, dass der Text als String vorliegt
    df = df[df[text_column].notna()]
    df[text_column] = df[text_column].astype(str)

    # Lade GloVe-Embeddings in ein Dictionary
    glove_embeddings = {}
    with open(glove_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            glove_embeddings[word] = vector

    # Datenaufteilung in Training- und Testsets
    X = df[text_column]
    y = df[label_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Oversampling zur Behandlung von Klassenungleichgewicht
    ros = RandomOverSampler(random_state=random_state)
    X_train_resampled, y_train_resampled = ros.fit_resample(X_train.values.reshape(-1, 1), y_train.values)
    X_train_resampled = X_train_resampled[:, 0]

    # Tokenisiere den Text mit Keras Tokenizer
    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_resampled)
    X_train_sequences = tokenizer.texts_to_sequences(X_train_resampled)
    X_test_sequences = tokenizer.texts_to_sequences(X_test)

    # Padding der Sequenzen
    X_train_padded = pad_sequences(X_train_sequences, padding='post', maxlen=max_seq_len)
    X_test_padded = pad_sequences(X_test_sequences, padding='post', maxlen=max_seq_len)

    # Funktion zur Umwandlung einer Sequenz in GloVe-Vektoren
    def get_glove_vectors(sequence, glove_embeddings, vector_size, max_seq_len):
        vectors = [glove_embeddings.get(tokenizer.index_word[idx], np.zeros(vector_size)) for idx in sequence if idx in tokenizer.index_word]
        
        # Sicherstellen, dass alle Sequenzen exakt `max_seq_len` lang sind
        if len(vectors) < max_seq_len:
            vectors.extend([np.zeros(vector_size)] * (max_seq_len - len(vectors)))
        else:
            vectors = vectors[:max_seq_len]
        
        return np.array(vectors, dtype=np.float32)

    # Wandle Sequenzen in Vektoren um
    X_train_vectors = np.array([get_glove_vectors(seq, glove_embeddings, vector_size, max_seq_len) for seq in X_train_padded], dtype=np.float32)
    X_test_vectors = np.array([get_glove_vectors(seq, glove_embeddings, vector_size, max_seq_len) for seq in X_test_padded], dtype=np.float32)

    return X_train_vectors, X_test_vectors, y_train_resampled, y_test, glove_embeddings, tokenizer


def vectorize_glove_test_data(df, text_column, label_column, glove_path, vector_size=200, max_seq_len=50, tokenizer=None):
    """
    Vectorizes text data using pre-trained GloVe embeddings.
    Each sentence is represented as a sequence of word vectors.

    Args:
        df (pd.DataFrame): The DataFrame containing the text and labels.
        text_column (str): The name of the column containing text data to vectorize.
        label_column (str): The name of the column containing target labels.
        glove_path (str): Path to the pre-trained GloVe embeddings file.
        vector_size (int, optional): The size of the GloVe vectors (default is 200).
        max_seq_len (int, optional): The maximum sequence length for padding (default is 50).
        tokenizer (Tokenizer, optional): The fitted tokenizer object from training. If None, a new tokenizer is created.

    Returns:
        train_vectors (np.ndarray): GloVe vectorized data.
        train_y (pd.Series): Labels of the training data.
        glove_embeddings (dict): The loaded GloVe embedding dictionary.
        tokenizer (Tokenizer): The fitted tokenizer.
    """
    # Entferne fehlende Werte und stelle sicher, dass der Text als String vorliegt
    df = df[df[text_column].notna()]
    df[text_column] = df[text_column].astype(str)

    # Lade GloVe-Embeddings in ein Wörterbuch
    glove_embeddings = {}
    with open(glove_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            glove_embeddings[word] = vector

    # Falls kein Tokenizer übergeben wurde, erstelle einen neuen
    if tokenizer is None:
        tokenizer = Tokenizer(oov_token="<OOV>")
        tokenizer.fit_on_texts(df[text_column])

    # Tokenisiere den Text
    X_sequences = tokenizer.texts_to_sequences(df[text_column])

    # Padding der Sequenzen
    X_padded = pad_sequences(X_sequences, padding='post', maxlen=max_seq_len)

    # Funktion zur Umwandlung einer Sequenz in GloVe-Vektoren
    def get_glove_vectors(sequence, glove_embeddings, vector_size, max_seq_len):
        vectors = [glove_embeddings.get(tokenizer.index_word.get(idx, ''), np.zeros(vector_size)) for idx in sequence]
        
        # Sicherstellen, dass alle Sequenzen exakt `max_seq_len` lang sind
        if len(vectors) < max_seq_len:
            vectors.extend([np.zeros(vector_size)] * (max_seq_len - len(vectors)))
        else:
            vectors = vectors[:max_seq_len]
        
        return np.array(vectors, dtype=np.float32)

    # Wandle Sequenzen in Vektoren um
    train_vectors = np.array([get_glove_vectors(seq, glove_embeddings, vector_size, max_seq_len) for seq in X_padded], dtype=np.float32)

    return train_vectors, df[label_column], glove_embeddings, tokenizer

def vectorize_glove_test_data_predict(df, text_column, label_column, glove_path, vector_size=200, max_seq_len=50, tokenizer=None):
    """
    Vectorizes text data using pre-trained GloVe embeddings.
    Each sentence is represented as a sequence of word vectors.

    Args:
        df (pd.DataFrame): The DataFrame containing the text and labels.
        text_column (str): The name of the column containing text data to vectorize.
        label_column (str): The name of the column containing target labels.
        glove_path (str): Path to the pre-trained GloVe embeddings file.
        vector_size (int, optional): The size of the GloVe vectors (default is 200).
        max_seq_len (int, optional): The maximum sequence length for padding (default is 50).
        tokenizer (Tokenizer, optional): The fitted tokenizer object from training. If None, a new tokenizer is created.

    Returns:
        train_vectors (np.ndarray): GloVe vectorized data.
        train_y (pd.Series): Labels of the training data.
        glove_embeddings (dict): The loaded GloVe embedding dictionary.
        tokenizer (Tokenizer): The fitted tokenizer.
    """
    # Entferne fehlende Werte und stelle sicher, dass der Text als String vorliegt
    df = df[df[text_column].notna()]
    df[text_column] = df[text_column].astype(str)

    # Lade GloVe-Embeddings in ein Wörterbuch
    glove_embeddings = {}
    with open(glove_path, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            glove_embeddings[word] = vector

    # Falls kein Tokenizer übergeben wurde, erstelle einen neuen
    if tokenizer is None:
        tokenizer = Tokenizer(oov_token="<OOV>")
        tokenizer.fit_on_texts(df[text_column])

    # Tokenisiere den Text
    X_sequences = tokenizer.texts_to_sequences(df[text_column])

    # Padding der Sequenzen
    X_padded = pad_sequences(X_sequences, padding='post', maxlen=max_seq_len)

    # Funktion zur Umwandlung einer Sequenz in GloVe-Vektoren
    def get_glove_vectors(sequence, glove_embeddings, vector_size, max_seq_len):
        vectors = [glove_embeddings.get(tokenizer.index_word.get(idx, ''), np.zeros(vector_size)) for idx in sequence]
        
        # Sicherstellen, dass alle Sequenzen exakt `max_seq_len` lang sind
        if len(vectors) < max_seq_len:
            vectors.extend([np.zeros(vector_size)] * (max_seq_len - len(vectors)))
        else:
            vectors = vectors[:max_seq_len]
        
        return np.array(vectors, dtype=np.float32)

    # Wandle Sequenzen in Vektoren um
    train_vectors = np.array([get_glove_vectors(seq, glove_embeddings, vector_size, max_seq_len) for seq in X_padded], dtype=np.float32)

    return train_vectors, df[text_column] ,df[label_column], glove_embeddings, tokenizer

