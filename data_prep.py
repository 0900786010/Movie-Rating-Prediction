import pandas as pd
import os

def load_movielens_100k(path="data/ml-100k/u.data"):
    # u.data file has no headers, so we manually name them
    df = pd.read_csv(path, sep="\t", header=None, names=["userId","movieId","rating","timestamp"])
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    df = load_movielens_100k("data/ml-100k/u.data")
    print("Loaded:", df.shape)

    # Save clean CSV
    df[["userId","movieId","rating"]].to_csv("data/ratings_clean.csv", index=False)

    print("Saved: data/ratings_clean.csv")
