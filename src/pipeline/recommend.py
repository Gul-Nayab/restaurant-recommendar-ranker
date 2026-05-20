from pathlib import Path

import joblib
import pandas as pd

from src.features.user_feature import add_user_features


FEATURES_PATH = Path("data/processed/restaurant_features_all_cities.csv")

MODEL_PATHS = {
    "logistic": Path("models/all_cities/logistic_model.pkl"),
    "xgboost": Path("models/all_cities/xgboost_model.pkl"),
}


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


def compute_model_score(model, X):
    probabilities = model.predict_proba(X)
    class_labels = model.classes_

    return probabilities @ class_labels


def generate_explanation(row, user_profile):
    reasons = []

    reasons.append(f"it is {row['distance']:.2f} miles away")

    if row["cuisine_match"] == 1:
        cuisines = ", ".join(user_profile.get("preferred_cuisines", []))
        reasons.append(f"matches your cuisine preferences: {cuisines}")

    if row["price_match"] == 1:
        reasons.append("matches your preferred price range")

    if row["dietary_match"] == 1 and user_profile.get("dietary_restrictions"):
        dietary = ", ".join(user_profile["dietary_restrictions"])
        reasons.append(f"supports your dietary restriction: {dietary}")

    reasons.append(
        f"has a {row['stars']} star rating with {int(row['review_count'])} reviews"
    )

    return "Recommended because " + ", ".join(reasons) + "."


def filter_by_location_fields(df, user_profile):
    filtered_df = df.copy()

    if "city" in user_profile and user_profile["city"]:
        city_df = filtered_df[
            filtered_df["city"].fillna("").str.lower() == user_profile["city"].lower()
        ]

        if not city_df.empty:
            filtered_df = city_df

    if "state" in user_profile and user_profile["state"]:
        state_df = filtered_df[
            filtered_df["state"].fillna("").str.lower() == user_profile["state"].lower()
        ]

        if not state_df.empty:
            filtered_df = state_df

    return filtered_df.copy()


def recommend(user_profile, model_name="xgboost", top_n=10):
    if model_name not in MODEL_PATHS:
        raise ValueError(f"Unknown model name: {model_name}")

    df = pd.read_csv(FEATURES_PATH)

    df = filter_by_location_fields(df, user_profile)

    df = add_user_features(df, user_profile)

    df = df[df["within_distance"] == 1].copy()

    if df.empty:
        return pd.DataFrame(
            columns=[
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
        )

    df = df.dropna(subset=FEATURE_COLUMNS)

    model = joblib.load(MODEL_PATHS[model_name])

    X = df[FEATURE_COLUMNS]

    df["score"] = compute_model_score(model, X)

    df = df.sort_values("score", ascending=False)

    df = df.drop_duplicates(
        subset=["name", "latitude", "longitude"],
        keep="first",
    )

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
