import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
import emoji
from nltk.corpus import stopwords, wordnet
import time
from pathlib import Path
from collections import Counter
from spellchecker import SpellChecker
from textblob import TextBlob, Word
from nltk.stem.snowball import SnowballStemmer, PorterStemmer
from nltk.stem import WordNetLemmatizer
from ftfy import fix_encoding
import spacy

from .shortcut_lists import shortcuts, shortforms, smileys

nlp = spacy.load("en_core_web_sm")

pd.set_option('display.max_colwidth', None)


def remove_special_characters(df, column_name):
    """
    Removes special characters from a specific column in a DataFrame.
    This function removes special characters such as HTML tags, mentions (e.g., @username), 
    and specific symbols (e.g., '/', '§', '&', '↝', '\'') from the specified column of the DataFrame. 
    Non-string values in the column are left unchanged.

    Args:
        df (pd.DataFrame): The DataFrame to process.
        column_name (str): The name of the column from which special characters should be removed.
    
    Returns:
        pd.DataFrame: The modified DataFrame with the special characters removed from the specified column.

    Raises:
        ValueError: If the specified column does not exist in the DataFrame.

    """
    # Check if the column exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

    # Compile the regex pattern
    pattern = re.compile(r"<.*?>|@\w+|[\/§&↝']")

    # Apply the regex substitution to the specified column
    df[column_name] = df[column_name].apply(
        lambda x: pattern.sub('', x) if isinstance(x, str) else x
    )
    return df


def remove_url_from_tweet(df, column_name):
    """
    Removes URLs from the specified column of a DataFrame.
    """
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")
    pattern = re.compile(r'https?://\S+|www\.\S+')
    df[column_name] = df[column_name].apply(lambda x: pattern.sub('', x))
    return df


def remove_punctuation(df, column_name):
    """
    Removes punctuation from the specified column of a DataFrame.
    """
    pattern = re.compile(r'[.:?!,\[\]/%&§{}]')
    df[column_name] = df[column_name].apply(lambda x: pattern.sub('', x))
    return df


def remove_digits(df, column_name):
    """
    Removes digits from the specified column of a DataFrame.
    """
    pattern = re.compile(r'\d')
    df[column_name] = df[column_name].apply(lambda x: pattern.sub('', x))
    return df


def expand_slang(text):
    """
    Expands slang terms in a text based on a predefined shortcuts dictionary.
    """
    new_text = []
    for w in text.split():
        if w.upper() in shortcuts:
            new_text.append(shortcuts[w.upper()])
        else:
            new_text.append(w)
    return ' '.join(new_text)


def expand_shortforms(text):
    """
    Expands common short forms in a text based on a predefined shortforms dictionary.
    """
    for shortform, fullform in shortforms.items():
        text = re.sub(re.escape(shortform), fullform, text)
    return text


def expand_shortcuts(df, column_name):
    """
    Expands slang terms and short forms in the specified column of a DataFrame.
    """
    df[column_name] = df[column_name].apply(expand_slang)
    df[column_name] = df[column_name].apply(expand_shortforms)
    return df


def remove_stop_words(df, column_name):
    """
    Removes stop words from the specified column of a DataFrame.
    """
    stop_words = set(stopwords.words('english'))
    df[column_name] = df[column_name].apply(
        lambda x: " ".join(w for w in x.split() if w.lower() not in stop_words))
    return df


def to_lowercase_if_string(text):
    """
    Converts a string to lowercase if the input is a string.
    """
    if isinstance(text, str):
        return text.lower()
    return text


def to_lowercase(df, column_name):
    """
    Converts all text in the specified column of a DataFrame to lowercase.
    """
    df[column_name] = df[column_name].fillna('').apply(to_lowercase_if_string)
    return df


def correct_misspelled_words_in_sentence(text):
    """
    Corrects misspelled words in a text using TextBlob.
    """
    words = text.split()
    corrected_text = []
    for word in words:
        text_blob = TextBlob(word)
        corrected_text.append(str(text_blob.correct()))
    return ' '.join(corrected_text)


def clean_misspelled_words(df, column_name):
    """
    Corrects misspelled words in the specified column of a DataFrame.
    """
    df[column_name] = df[column_name].apply(correct_misspelled_words_in_sentence)
    return df


def replace_emoji_in_sentence(text):
    """
    Replaces emojis in a text with their descriptive names.
    """
    if isinstance(text, str):
        new_text = emoji.demojize(text, delimiters=("__", "__"))
        new_text = new_text.replace("__", " ").replace("_", " ")
        return new_text
    return text


def replace_emojis(df, column_name):
    """
    Replaces emojis in the specified column of a DataFrame with their descriptive names.
    """
    df[column_name] = df[column_name].apply(replace_emoji_in_sentence)
    return df


def remove_emoji_in_sentence(text):
    """
    Removes all emojis from a text.
    """
    emoji_pattern = re.compile("[" + u"\U0001F600-\U0001F64F" +
                               u"\U0001F300-\U0001F5FF" +
                               u"\U0001F680-\U0001F6FF" +
                               u"\U0001F1E0-\U0001F1FF" +
                               u"\U00002702-\U000027B0" +
                               u"\U00002FC2-\U0001F251" + "]+", flags=re.UNICODE)
    if isinstance(text, str):
        return emoji_pattern.sub(r'', text)
    return text


def remove_emojis(df, column_name):
    """
    Removes emojis from the specified column of a DataFrame.
    """
    df[column_name] = df[column_name].apply(remove_emoji_in_sentence)
    return df


def get_emojis(text):
    """
    Extracts emojis from a text and returns them as a comma-separated string.
    """
    emoji_list = []
    for word in text.split():
        for char in word:
            if emoji.is_emoji(char):
                emoji_list.append(emoji.demojize(char))
    return ','.join(emoji_list)


def extract_emojis(df, column_name):
    """
    Extracts emojis from the specified column of a DataFrame and stores them in a new column.
    """
    df['emojis'] = df[column_name].apply(get_emojis)
    return df


def remove_hashtag_sign_from_tweet(text):
    """
    Removes the hashtag symbol (#) from hashtags in a text.
    """
    new_text = []
    for word in text.split():
        if word.startswith('#'):
            new_text.append(word[1:])
        else:
            new_text.append(word)
    return ' '.join(new_text)


def handle_hashtags(df, column_name):
    """
    Extracts hashtags and removes the hashtag symbol from the specified column of a DataFrame.
    """
    df['hashtags'] = [re.findall(r'#\w+', x) if re.findall(r'#\w+', x) else [] for x in df[column_name]]
    df[column_name] = df[column_name].str.replace('#', '', regex=False)
    return df


def handle_userhandles(df, column_name):
    """
    Counts and removes user handles (@user) from the specified column of a DataFrame.
    """
    df['user_handle'] = df[column_name].str.count('@user')
    df[column_name] = df[column_name].str.replace('@user', '', case=False)
    return df


def create_word_counter(col):
    """
    Creates a word frequency counter for a given column of text.
    """
    cnt = Counter()
    for text in col.values:
        for word in text.split():
            cnt[word] += 1
    return cnt


def remove_freqwords(text, freqwords):
    """
    Removes the most frequent words from a given text.
    """
    return " ".join([word for word in str(text).split() if word not in freqwords])


def remove_most_frequent_words(df, column_name):
    """
    Removes the most frequent words from the specified column of a DataFrame.
    """
    counter = create_word_counter(df[column_name])
    freqwords = set([w for (w, wc) in counter.most_common(10)])
    df[column_name] = df[column_name].apply(lambda text: remove_freqwords(text, freqwords))
    return df


def find_words(col):
    """
    Finds and counts words with exactly 7 characters in a given column.
    """
    cnt = Counter()
    for text in col.values:
        for word in text.split():
            word = re.sub('[^A-Za-z]+', '', word)
            if len(word) == 7:
                cnt[word] += 1
    return cnt


def spacy_lemmatize(text):
    """
    Lemmatizes a given text using SpaCy.
    """
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])


def lemmatize(df, column_name):
    """
    Lemmatizes all text in the specified column of a DataFrame.
    """
    df[column_name] = df[column_name].apply(lambda text: spacy_lemmatize(text))
    return df


def remove_least_freqwords(text, least_freqwords):
    """
    Removes the least frequent words from a given text.
    """
    return " ".join([word for word in str(text).split() if word not in least_freqwords])


def remove_least_frequent_words(df, column_name):
    """
    Removes the least frequent words from the specified column of a DataFrame.
    """
    min_threshold = 3
    counter = create_word_counter(df[column_name])
    filtered_counter = {k: v for k, v in counter.items() if v <= min_threshold}
    least_freqwords = set(filtered_counter.keys())
    df[column_name] = df[column_name].apply(lambda text: remove_least_freqwords(text, least_freqwords))
    return df


def remove_duplicates(df, column_name):
    """
    Removes duplicate rows based on the specified column of a DataFrame.
    """
    df = df.drop_duplicates(subset=[column_name])
    return df


def remove_word_from_column(df, column_name, word):
    """
    Removes a specified word from the values in a column of a DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column to process.
        word (str): The word to remove from the column values.
    
    Returns:
        pd.DataFrame: A DataFrame with the specified word removed from the column.
    """
    df[column_name] = df[column_name].str.replace(rf'\b{word}\b', '', regex=True).str.strip()
    return df


import pandas as pd


def remove_na_from_column(df, column_name):
    """
    Removes rows from a DataFrame where the specified column contains NaN or missing values.

    Args:
        df (pd.DataFrame): The DataFrame to clean.
        column_name (str): The name of the column to check for NaN or missing values.

    Returns:
        pd.DataFrame: The cleaned DataFrame without NaN or missing values in the specified column.
    """
    return df[df[column_name].notna()]


def replace_smileys(text):
    """
    Replace text smiley like 'xD' in a text with the meaning based on a predefined smiley dictionary.
    """
    new_text = []
    for w in text.split():
        if w.upper() in smileys:
            new_text.append(smileys[w.upper()])
        else:
            new_text.append(w)
    return ' '.join(new_text)


def replace_text_smileys(df, column_name):
    """
    Converts all text smileys in the specified column of a DataFrame to the textual meaning.
    """
    df[column_name] = df[column_name].fillna('').apply(replace_smileys)
    return df


from ekphrasis.classes.segmenter import Segmenter


def segment_text_cases(text):
    """
    Separates CamelCase and pascalCase words into strings.

    # Code reference: https://github.com/cbaziotis/ekphrasis
    """
    # segmenter for CamelCase and pascalCase
    seg = Segmenter()

    new_text = []
    for w in text.split():
        new_word = seg.segment(w)

        new_text.append(new_word)
    return ' '.join(new_text)

def segment_text_english(text):
    """
    Separates english words into strings according to a provided corpus.

    # Code reference: https://github.com/cbaziotis/ekphrasis
    """
    # segmenter using the word statistics from english Wikipedia
    seg_eng = Segmenter(corpus="english")

    new_text = []
    for w in text.split():
        new_word = seg_eng.segment(w)

        new_text.append(new_word)
    return ' '.join(new_text)


def segment_text_twitter(text):
    """
    Separates twitter words into strings according to a provided corpus.

    # Code reference: https://github.com/cbaziotis/ekphrasis
    """
    # segmenter using the word statistics from Twitter
    seg_tw = Segmenter(corpus="twitter")

    new_text = []
    for w in text.split():
        new_word = seg_tw.segment(new_word)

        new_text.append(new_word)
    return ' '.join(new_text)


def segment_tweets(df, column_name):
    """
    Separates all text in the specified column of a DataFrame according to a provided corpus.
    """
    df[column_name] = df[column_name].fillna('').apply(segment_text_cases)
    df[column_name] = df[column_name].fillna('').apply(segment_text_english)
    df[column_name] = df[column_name].fillna('').apply(segment_text_twitter)
    return df
