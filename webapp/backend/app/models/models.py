import joblib
import pickle
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.models import load_model

# SVM
vectorizer_svm_tfidf = joblib.load("models/svc/tfidf_vectorizer_svc.joblib")
model_svm = joblib.load("models/svc/model_svc.joblib")

# ENSEMBLE
tfidf_vectorizer_ensemble = joblib.load("models/ensemble/tfidf_vectorizer_for_brf.joblib")
model_ensemble = joblib.load("models/ensemble/tfidf_balancedrandomforest.joblib")

# NAIVE BAYES
vectorizer_nb_tfidf = joblib.load("models/nb/vectorizer_nb_tfidf.joblib")
model_nb = joblib.load("models/nb/model_nb_tfidf_comp.joblib")

# RNN-LSTM
threshold_lstm = 0.35
model_lstm = load_model("models/lstm/model_lstm_17.keras")
with open("models/lstm/tokenizer_lstm.json", "r", encoding="utf-8") as f:
    tokenizer_data = f.read()
    tokenizer_lstm = tokenizer_from_json(tokenizer_data)
lstm_glove_embeddings = {}
with open("models/lstm/glove.6B.200d.txt", encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        lstm_glove_embeddings[word] = vector

# RNN-GRU
threshold_gru = 0.35
max_len_gru = 40
model_gru = load_model("models/gru/gru-model_mixed-dataset.keras")
with open("models/gru/tokenizer_mixed-dataset.pkl", 'rb') as f:
    tokenizer_gru = pickle.load(f)

# BERT
PATH_BERT_TUNED = f"models/bert/bert_mixed_imran"
tokenizer_bert = BertTokenizer.from_pretrained(PATH_BERT_TUNED, local_files_only=True)
model_bert = AutoModelForSequenceClassification.from_pretrained(PATH_BERT_TUNED, local_files_only=True)

# ROBERTA
PATH_ROBERTA_TUNED = f"models/roberta/roberta_hate_mixed_cleaned"
tokenizer_roberta = AutoTokenizer.from_pretrained(PATH_ROBERTA_TUNED, local_files_only=True)
model_roberta = AutoModelForSequenceClassification.from_pretrained(PATH_ROBERTA_TUNED, local_files_only=True)


# spezielle LSTM-Vektorisierungsfunktion für einzelnen Tweet
def vectorize_glove_test_data_predict(text, vector_size=200, max_seq_len=50, tokenizer=None):
    """
    Vektorisiert einen einzelnen Textstring mit vortrainierten GloVe-Embeddings.

    Args:
        text (str): Der Eingabetext, der vektorisiert werden soll.
        glove_path (str): Pfad zur GloVe-Embeddings-Datei.
        vector_size (int, optional): Größe der GloVe-Vektoren (Standard: 200).
        max_seq_len (int, optional): Maximale Sequenzlänge für Padding (Standard: 50).
        tokenizer (Tokenizer, optional): Der trainierte Tokenizer. Falls None, wird ein neuer erstellt.

    Returns:
        np.ndarray: Ein 3D-Array mit der Form `(1, max_seq_len, vector_size)`, das direkt für LSTM nutzbar ist.
    """

    if tokenizer is None:
        raise ValueError("Ein trainierter Tokenizer muss übergeben werden!")

    X_sequence = tokenizer.texts_to_sequences([text])  # Text in Sequenz umwandeln (Liste mit 1 Element)

    X_padded = pad_sequences(X_sequence, padding='post', maxlen=max_seq_len)

    def get_glove_vectors(sequence, glove_embeddings, vector_size, max_seq_len):
        vectors = [glove_embeddings.get(tokenizer.index_word.get(idx, ''), np.zeros(vector_size)) for idx in sequence]

        # Padding sicherstellen
        if len(vectors) < max_seq_len:
            vectors.extend([np.zeros(vector_size)] * (max_seq_len - len(vectors)))
        else:
            vectors = vectors[:max_seq_len]

        return np.array(vectors, dtype=np.float32)

    X_vectorized = np.array(get_glove_vectors(X_padded[0], lstm_glove_embeddings, vector_size, max_seq_len),
                            dtype=np.float32)

    return np.expand_dims(X_vectorized, axis=0)
