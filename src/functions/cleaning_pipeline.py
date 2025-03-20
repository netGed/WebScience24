from src.functions.clean_data_generic_functions import to_lowercase, expand_shortcuts, handle_userhandles, \
    handle_hashtags, extract_emojis, replace_emojis, replace_text_smileys, remove_url_from_tweet, remove_punctuation, \
    remove_special_characters, remove_digits, remove_word_from_column, lemmatize, remove_stop_words, \
    remove_most_frequent_words, remove_least_frequent_words, remove_duplicates, remove_na_from_column
from ftfy import fix_encoding

def clean_dataframe(df, col_to_clean, simple_cleaning_only, drop_duplicate, deep_learning):
    """
    Cleanes and preprocesses a text column of a DataFrame for Natural Language Processing in Machine Learning and Deep Learning models.
   Removes rows from a DataFrame where the specified column contains NaN or missing values.

   Args:
       df (pd.DataFrame): The DataFrame to clean.
       col_to_clean (str): The name of the column to clean and preprocess.
       simple_cleaning_only (bool): If True, only Encoding will be fixed.
       drop_duplicate (bool): Decides if rows can be dropped (should not be done for test DataFrames)
       deep_learning (bool): If true, preprocessing steps that could worsen deep learning models like lemmatization are not done.

   Returns:
       pd.DataFrame: The cleaned DataFrame.
   """
    i = 1
    count = 20
    df_cleaned = df.copy()

    if drop_duplicate:
        df_cleaned.drop_duplicates(inplace=True)
    df_cleaned[col_to_clean] = df_cleaned[col_to_clean].apply(fix_encoding)

    if simple_cleaning_only:
        return df_cleaned

    new_col_name = col_to_clean + '_cleaned'
    df_cleaned[new_col_name] = df_cleaned[col_to_clean]

    print("Start Cleaning")

    print(f"--- Cleaning Step {i}/{count}: to_lowercase")
    df_cleaned = to_lowercase(df_cleaned, 'tweet_cleaned')
    i = i + 1

    print(f"--- Cleaning Step {i}/{count}: expand_shortcuts")
    df_cleaned = expand_shortcuts(df_cleaned, 'tweet_cleaned')
    i = i + 1

    print(f"--- Cleaning Step {i}/{count}: remove_negations - SKIP (not implemented)")
    # # df_cleaned = remove_negations(df_cleaned)
    i = i + 1

    print(f"--- Cleaning Step {i}/{count}: handle_userhandles")
    df_cleaned = handle_userhandles(df_cleaned, 'tweet_cleaned')
    i = i + 1

    print(f"--- Cleaning Step {i}/{count}: handle_hashtags")
    df_cleaned = handle_hashtags(df_cleaned, 'tweet_cleaned')
    i = i + 1

    print(f"--- Cleaning Step {i}/{count}: extract_emojis")
    df_cleaned = extract_emojis(df_cleaned, 'tweet_cleaned')
    i = i + 1

    print(f"--- Cleaning Step {i}/{count}: replace_emojis")
    df_cleaned = replace_emojis(df_cleaned, 'tweet_cleaned')
    i = i + 1

    print(f"--- Cleaning Step {i}/{count}: replace_smileys")
    df_cleaned = replace_text_smileys(df_cleaned, 'tweet_cleaned')
    i = i + 1

    print(f"--- Cleaning Step {i}/{count}: remove_emojis - SKIP")
    # df_cleaned = remove_emojis(df_cleaned)
    i = i + 1

    print(f"--- Cleaning Step {i}/{count}: remove_url_from_tweet")
    df_cleaned = remove_url_from_tweet(df_cleaned, 'tweet_cleaned')
    i = i + 1

    print(f"--- Cleaning Step {i}/{count}: remove_punctuation")
    df_cleaned = remove_punctuation(df_cleaned, 'tweet_cleaned')
    i = i + 1

    print(f"--- Cleaning Step {i}/{count}: remove_special_characters")
    df_cleaned = remove_special_characters(df_cleaned, 'tweet_cleaned')
    i = i + 1

    print(f"--- Cleaning Step {i}/{count}: remove_digis")
    df_cleaned = remove_digits(df_cleaned, 'tweet_cleaned')
    i = i + 1

    print(f"--- Cleaning Step {i}/{count}: remove_word_from_column: amp")
    df_cleaned = remove_word_from_column(df=df_cleaned, column_name="tweet_cleaned", word="amp")
    i = i + 1

    # skip cleaning/text-preprocessing functions that worsen deep-learning model performance
    if deep_learning:
        print(f"--- Cleaning Step {i}/{count}: lemmatize - SKIP")
        # df_cleaned = lemmatize(df_cleaned, 'tweet_cleaned')
        i = i + 1

        print(f"--- Cleaning Step {i}/{count}: remove_stop_words - SKIP")
        # df_cleaned = remove_stop_words(df_cleaned, 'tweet_cleaned')
        i = i + 1

        print(f"--- Cleaning Step {i}/{count}: remove_most_frequent_words - SKIP")
        # df_cleaned = remove_most_frequent_words(df_cleaned, 'tweet_cleaned')
        i = i + 1

        print(f"--- Cleaning Step {i}/{count}: remove_least_frequent_words - SKIP")
        # df_cleaned = remove_least_frequent_words(df_cleaned, 'tweet_cleaned')
        i = i + 1
    else:
        print(f"--- Cleaning Step {i}/{count}: lemmatize")
        df_cleaned = lemmatize(df_cleaned, 'tweet_cleaned')
        i = i + 1

        print(f"--- Cleaning Step {i}/{count}: remove_stop_words")
        df_cleaned = remove_stop_words(df_cleaned, 'tweet_cleaned')
        i = i + 1

        print(f"--- Cleaning Step {i}/{count}: remove_most_frequent_words")
        df_cleaned = remove_most_frequent_words(df_cleaned, 'tweet_cleaned')
        i = i + 1

        print(f"--- Cleaning Step {i}/{count}: remove_least_frequent_words")
        df_cleaned = remove_least_frequent_words(df_cleaned, 'tweet_cleaned')
        i = i + 1

    if drop_duplicate:
        print(f"--- Cleaning Step {i}/{count}: remove_duplicates")
        df_cleaned = remove_duplicates(df_cleaned, 'tweet_cleaned')
    else:
        print(f"--- Cleaning Step {i}/{count}: remove_duplicates - SKIP")
        # df_cleaned = remove_duplicates(df_cleaned, 'tweet_cleaned')
    i = i + 1

    print(f"--- Cleaning Step {i}/{count}: remove_nans")
    df_cleaned = remove_na_from_column(df=df_cleaned, column_name="tweet_cleaned")

    print("All Cleaning done")

    return df_cleaned
