import pandas as pd


def get_data_evaluation():
    df = pd.read_csv("data/evaluation_dataset.csv", delimiter=",")
    df = df[["id", "tweet", "label_manual"]]
    df = df.rename(columns={'label_manual': 'label'})
    df = df.sample(frac=1)
    return df.to_dict(orient="records")


def get_data_old(count: int = 1):
    df = pd.read_csv("data/old_test.csv", delimiter=",")
    df_rand = df.sample(n=count)
    return df_rand.to_dict(orient="records")


def get_data_new(count: int = 1):
    df = pd.read_csv("data/new_test.csv", delimiter=",")
    df_rand = df.sample(n=count)
    return df_rand.to_dict(orient="records")


def get_data_mixed(count: int = 1):
    df = pd.read_csv("data/mixed_test.csv", delimiter=",")
    df_rand = df.sample(n=count)
    return df_rand.to_dict(orient="records")
