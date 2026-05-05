from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from src.features.user_feature import add_user_features


FEATURES_PATH = Path("data/processed/restaurant_features_philadelphia.csv")
MODEL_PATH = Path("models/logistic_model.pkl")


TRAINING_USER_PROFILE = {
    "latitude": 39.95,
    "longitude": -75.16,
    "max_distance": 10,
    "preferred_cuisines": ["mexican", "vegan"],
    "dietary_restrictions": ["vegan"],
    "price_range": [1, 2],
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


def train_model():
    df = pd.read_csv(FEATURES_PATH)

    df = add_user_features(df, TRAINING_USER_PROFILE)
    df = create_labels(df)

    df = df.dropna(subset=FEATURE_COLUMNS + ["label"])

    X = df[FEATURE_COLUMNS]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    print("Training complete.")
    print(classification_report(y_test, predictions))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
