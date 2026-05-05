from pathlib import Path

import joblib
import pandas as pd

from src.features.user_feature import add_user_features


FEATURES_PATH = Path("data/processed/restaurant_features_philadelphia.csv")


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


MODEL_PATHS = {
    "logistic": Path("models/logistic_model.pkl"),
    "xgboost": Path("models/xgboost_model.pkl"),
}


def compute_model_score(model, X):
    probabilities = model.predict_proba(X)

    class_labels = model.classes_

    # Expected relevance score:
    # P(class 0)*0 + P(class 1)*1 + ... + P(class 8)*8
    scores = probabilities @ class_labels

    return scores


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
        reasons.append("has a high Yelp rating")

    if not reasons:
        return "Recommended based on overall model score."

    return "Recommended because it " + ", ".join(reasons) + "."


def recommend(user_profile, model_name="xgboost", top_n=10):
    if model_name not in MODEL_PATHS:
        raise ValueError(f"Unknown model name: {model_name}")

    df = pd.read_csv(FEATURES_PATH)

    df = add_user_features(df, user_profile)

    # Hard filter by distance before ranking
    df = df[df["within_distance"] == 1].copy()

    df = df.dropna(subset=FEATURE_COLUMNS)

    model = joblib.load(MODEL_PATHS[model_name])

    X = df[FEATURE_COLUMNS]

    df["score"] = compute_model_score(model, X)

    df = df.sort_values("score", ascending=False)

    results = df.head(top_n).copy()

    results["explanation"] = results.apply(
        lambda row: generate_explanation(row, user_profile),
        axis=1,
    )

    return results[
        [
            "name",
            "city",
            "state",
            "stars",
            "review_count",
            "price_level",
            "distance",
            "categories",
            "score",
            "explanation",
        ]
    ]
