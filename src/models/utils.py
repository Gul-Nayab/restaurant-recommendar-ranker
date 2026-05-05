from pathlib import Path
import json
import numpy as np
import pandas as pd

from src.features.user_feature import add_user_features


FEATURES_PATH = Path("data/processed/restaurant_features_philadelphia.csv")
USER_PROFILES_PATH = Path("data/processed/user_profiles_philadelphia.json")
MODEL_PATH = Path("models/logistic_model.pkl")


FEATURE_COLUMNS = [
    "stars",
    "log_review_count",
    "price_level",
    "distance",
    "within_distance",
    "cuisine_match",
    "price_match",
    "dietary_match",
]


def load_user_profiles():
    with open(USER_PROFILES_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def create_labels(df, add_noise=True):
    df = df.copy()

    df["label_score"] = (
        df["within_distance"] * 2
        + df["cuisine_match"] * 2
        + df["price_match"] * 1
        + df["dietary_match"] * 2
        + (df["stars"] >= 4).astype(int)
    )

    return df


def build_training_data(restaurants_df, user_profiles):
    training_rows = []

    for user in user_profiles:
        user_df = add_user_features(restaurants_df, user)
        user_df = create_labels(user_df, add_noise=True)
        user_df["user_id"] = user["user_id"]

        training_rows.append(user_df)

    return pd.concat(training_rows, ignore_index=True)
