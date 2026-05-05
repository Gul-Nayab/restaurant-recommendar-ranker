from pathlib import Path
import time

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier

from src.models.utils import load_user_profiles, build_training_data


FEATURES_PATH = Path("data/processed/restaurant_features_philadelphia.csv")
RESULTS_PATH = Path("data/processed/model_comparison_results.csv")


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


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    start_train = time.perf_counter()
    model.fit(X_train, y_train)
    train_time = time.perf_counter() - start_train

    start_inference = time.perf_counter()
    predictions = model.predict(X_test)
    inference_time = time.perf_counter() - start_inference

    return {
        "model": name,
        "train_time_seconds": train_time,
        "inference_time_seconds": inference_time,
        "accuracy": accuracy_score(y_test, predictions),
        "precision_weighted": precision_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "recall_weighted": recall_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
        "f1_weighted": f1_score(
            y_test, predictions, average="weighted", zero_division=0
        ),
    }


def main():
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

    models = [
        (
            "Logistic Regression",
            Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("logreg", LogisticRegression(max_iter=3000)),
                ]
            ),
            Path("models/logistic_model.pkl"),
        ),
        (
            "XGBoost",
            XGBClassifier(
                n_estimators=150,
                max_depth=4,
                learning_rate=0.1,
                objective="multi:softprob",
                eval_metric="mlogloss",
                random_state=42,
            ),
            Path("models/xgboost_model.pkl"),
        ),
    ]

    results = []

    for name, model, model_path in models:
        result = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results.append(result)

        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

    results_df = pd.DataFrame(results)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)

    print("\nModel Comparison Results:")
    print(results_df)
    print(f"\nSaved results to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
