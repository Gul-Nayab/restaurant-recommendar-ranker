from pathlib import Path
import json
import time

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

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


def create_labels(df):
    df = df.copy()

    df["label"] = (
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
        user_df = create_labels(user_df)
        user_df["user_id"] = user["user_id"]

        training_rows.append(user_df)

    training_df = pd.concat(training_rows, ignore_index=True)

    return training_df


def train_logistic():
    restaurants_df = pd.read_csv(FEATURES_PATH)
    user_profiles = load_user_profiles()

    training_df = build_training_data(restaurants_df, user_profiles)

    training_df = training_df.dropna(subset=FEATURE_COLUMNS + ["label"])

    X = training_df[FEATURE_COLUMNS]
    y = training_df["label"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = LogisticRegression(max_iter=1000)

    start_time = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start_time

    start_time = time.perf_counter()
    predictions = model.predict(X_test)
    inference_time = time.perf_counter() - start_time

    print("Logistic Regression training complete.")
    print(f"Training time: {train_time:.4f} seconds")
    print(f"Inference time: {inference_time:.4f} seconds")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.4f}")
    print(f"F1 weighted: {f1_score(y_test, predictions, average='weighted'):.4f}")
    print()
    print(classification_report(y_test, predictions))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    train_logistic()
