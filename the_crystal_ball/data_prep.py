# standard library
import os

# external
import pandas as pd

# interal
from .configuration import config


def create_features(df_input, **meta):
    input_article = "Regenerative_agriculture"  # meta['input_article']

    df_features = pd.DataFrame()

    #     Filtering
    df_filtered = df_input.loc[df_input["article"] == input_article]

    df_features["ds"] = df_filtered["timestamp"]
    # df_features["y"] = df_filtered["views"]
    #     Scaling
    #     Resampling and Interpolation
    #     Power Transforms to remove noise (e.g. square root or log transform)
    #     Moving Average Smoothing
    df_features["y"] = df_filtered["views"].rolling(7).mean()

    return df_features.dropna()


def store_features(df_features):
    filepath = os.path.join(
        config.ROOT_PATH,
        "data",
        "features.csv",
    )
    print(f"writing features to {filepath}")
    df_features.to_csv(filepath, index=False)
    return filepath


def load_features(filename):
    filepath = os.path.join(
        config.ROOT_PATH,
        "data",
        "features.csv",
    )
    print(f"reading features from {filepath}")
    df = pd.read_csv(filepath)
    df["ds"] = df["ds"].map(pd.Timestamp)
    return df
