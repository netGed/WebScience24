import pandas as pd


def read_data():
    df = pd.read_csv("data/evaluation_dataset.csv", delimiter=";")
    return df.to_dict(orient="records")

