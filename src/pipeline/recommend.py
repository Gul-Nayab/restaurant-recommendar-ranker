from pathlib import Path

import joblib
import pandas as pd

from src.features.user_feature import add_user_features


FEATURES_PATH = Path("data/processed/restaurant_features_philadelphia.csv")
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


def generate_explanation(row, user_profile):
    reasons = []

    if row["within_distance"] == 1:
        reasons.append(f"within {user_profile['max_distance']} miles")

    if row["cuisine_match"] == 1:
        reasons.append("matches preferred cuisine")

    if row["price_match"] == 1:
        reasons.append("matches price preference")

    if row["dietary_match"] == 1 and user_profile.get("dietary_restrictions"):
        reasons.append(
            "matches dietary restriction: "
            + ", ".join(user_profile["dietary_restrictions"])
        )

    if row["stars"] >= 4:
        reasons.append("high Yelp rating")

    if not reasons:
        return "Recommended based on overall model score."

    return "Recommended because it " + ", ".join(reasons) + "."


def recommend(user_profile, top_n=10):
    df = pd.read_csv(FEATURES_PATH)

    df = add_user_features(df, user_profile)

    # Optional hard filter: only consider restaurants within max distance
    df = df[df["within_distance"] == 1].copy()

    df = df.dropna(subset=FEATURE_COLUMNS)

    model = joblib.load(MODEL_PATH)

    X = df[FEATURE_COLUMNS]

    if hasattr(model, "predict_proba"):
        df["score"] = model.predict_proba(X).max(axis=1)
    else:
        df["score"] = model.predict(X)

    df = df.sort_values("score", ascending=False)

    results = df.head(top_n).copy()

    results["explanation"] = results.apply(
        lambda row: generate_explanation(row, user_profile),
        axis=1,
    )

    return results[
        [
            "name",
            "stars",
            "review_count",
            "price_level",
            "distance",
            "categories",
            "score",
            "explanation",
        ]
    ]
