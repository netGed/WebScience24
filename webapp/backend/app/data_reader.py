import pandas as pd


def read_data():
    df = pd.read_csv("data/evaluation_dataset.csv", delimiter=";")
    df = df.sample(frac=1, random_state=42)
    return df.to_dict(orient="records")


def get_random_data():
    df = pd.read_csv("data/test.csv", delimiter=",")
    df_rand = df.sample(n=1)
    return df_rand.to_dict(orient="records")
