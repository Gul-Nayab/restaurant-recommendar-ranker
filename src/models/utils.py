import json

import numpy as np
import pandas as pd

from src.features.user_feature import add_user_features


def load_user_profiles(user_profiles_path):
    with open(user_profiles_path, "r", encoding="utf-8") as file:
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

    if add_noise:
        noise = np.random.normal(loc=0, scale=0.25, size=len(df))
        df["label_score"] = df["label_score"] + noise

    df["label"] = df["label_score"].round().clip(0, 8).astype(int)

    return df


def filter_candidates_for_user(restaurants_df, user):
    candidate_df = restaurants_df.copy()

    if "city" in user and user["city"] and "city" in candidate_df.columns:
        city_df = candidate_df[
            candidate_df["city"].fillna("").str.lower() == user["city"].lower()
        ]

        if not city_df.empty:
            candidate_df = city_df

    if "state" in user and user["state"] and "state" in candidate_df.columns:
        state_df = candidate_df[
            candidate_df["state"].fillna("").str.lower() == user["state"].lower()
        ]

        if not state_df.empty:
            candidate_df = state_df

    return candidate_df.copy()


def build_training_data(restaurants_df, user_profiles, add_noise=True):
    training_rows = []

    for user in user_profiles:
        candidate_df = filter_candidates_for_user(restaurants_df, user)

        user_df = add_user_features(candidate_df, user)
        user_df = create_labels(user_df, add_noise=add_noise)
        user_df["user_id"] = user["user_id"]

        training_rows.append(user_df)

    return pd.concat(training_rows, ignore_index=True)
