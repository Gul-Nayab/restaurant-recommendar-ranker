from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.features.user_feature import add_user_features
from src.models.utils import (
    load_user_profiles,
    create_labels,
    filter_candidates_for_user,
)


FEATURES_PATH = Path("data/processed/restaurant_features_all_cities.csv")
USER_PROFILES_PATH = Path("data/processed/user_profiles_all_cities.json")

MODEL_PATHS = {
    "logistic": Path("models/all_cities/logistic_model.pkl"),
    "xgboost": Path("models/all_cities/xgboost_model.pkl"),
}

DETAIL_RESULTS_PATH = Path(
    "data/processed/evaluation/ranking_metrics_by_user_all_cities.csv"
)
SUMMARY_RESULTS_PATH = Path(
    "data/processed/evaluation/ranking_metrics_summary_all_cities.csv"
)


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


# Labels range from 0 to 8.
# A label >= 6 means the restaurant is a strong match.
RELEVANCE_THRESHOLD = 6

K_VALUES = [5, 10]


def precision_at_k(labels, k, relevance_threshold=RELEVANCE_THRESHOLD):
    labels = np.asarray(labels)

    if len(labels) == 0:
        return np.nan

    top_k = labels[:k]
    relevant_count = np.sum(top_k >= relevance_threshold)

    return relevant_count / min(k, len(top_k))


def dcg_at_k(labels, k):
    labels = np.asarray(labels[:k], dtype=float)

    if len(labels) == 0:
        return 0.0

    ranks = np.arange(2, len(labels) + 2)
    gains = (2**labels - 1) / np.log2(ranks)

    return np.sum(gains)


def ndcg_at_k(labels, k):
    labels = np.asarray(labels, dtype=float)

    if len(labels) == 0:
        return np.nan

    actual_dcg = dcg_at_k(labels, k)
    ideal_labels = np.sort(labels)[::-1]
    ideal_dcg = dcg_at_k(ideal_labels, k)

    if ideal_dcg == 0:
        return np.nan

    return actual_dcg / ideal_dcg


def average_precision(labels, relevance_threshold=RELEVANCE_THRESHOLD):
    labels = np.asarray(labels)
    relevant = labels >= relevance_threshold

    total_relevant = np.sum(relevant)

    if total_relevant == 0:
        return np.nan

    precision_values = []
    hits = 0

    for rank, is_relevant in enumerate(relevant, start=1):
        if is_relevant:
            hits += 1
            precision_values.append(hits / rank)

    return np.sum(precision_values) / total_relevant


def compute_model_score(model, X):
    probabilities = model.predict_proba(X)
    class_labels = model.classes_

    # Expected relevance score:
    # P(label 0)*0 + P(label 1)*1 + ... + P(label 8)*8
    return probabilities @ class_labels


def evaluate_profile(model_name, model, restaurants_df, user_profile):
    candidate_df = filter_candidates_for_user(restaurants_df, user_profile)

    user_df = add_user_features(candidate_df, user_profile)

    # Use clean labels for evaluation.
    # Training can use noisy labels, but evaluation should use stable relevance labels.
    user_df = create_labels(user_df, add_noise=False)

    # Match recommendation pipeline: only rank restaurants within distance.
    user_df = user_df[user_df["within_distance"] == 1].copy()

    if user_df.empty:
        return {
            "model": model_name,
            "user_id": user_profile.get("user_id"),
            "city": user_profile.get("city"),
            "state": user_profile.get("state"),
            "num_candidates": 0,
            "num_relevant": 0,
            "precision_at_5": np.nan,
            "precision_at_10": np.nan,
            "ndcg_at_5": np.nan,
            "ndcg_at_10": np.nan,
            "average_precision": np.nan,
        }

    user_df = user_df.dropna(subset=FEATURE_COLUMNS + ["label"])

    if user_df.empty:
        return {
            "model": model_name,
            "user_id": user_profile.get("user_id"),
            "city": user_profile.get("city"),
            "state": user_profile.get("state"),
            "num_candidates": 0,
            "num_relevant": 0,
            "precision_at_5": np.nan,
            "precision_at_10": np.nan,
            "ndcg_at_5": np.nan,
            "ndcg_at_10": np.nan,
            "average_precision": np.nan,
        }

    X = user_df[FEATURE_COLUMNS]

    user_df["score"] = compute_model_score(model, X)

    user_df = user_df.sort_values("score", ascending=False)

    user_df = user_df.drop_duplicates(
        subset=["name", "latitude", "longitude"],
        keep="first",
    )

    labels = user_df["label"].to_numpy()

    result = {
        "model": model_name,
        "user_id": user_profile.get("user_id"),
        "city": user_profile.get("city"),
        "state": user_profile.get("state"),
        "num_candidates": len(user_df),
        "num_relevant": int(np.sum(labels >= RELEVANCE_THRESHOLD)),
        "average_precision": average_precision(labels),
    }

    for k in K_VALUES:
        result[f"precision_at_{k}"] = precision_at_k(labels, k)
        result[f"ndcg_at_{k}"] = ndcg_at_k(labels, k)

    return result


def main():
    print("Loading all-cities restaurant features...")
    restaurants_df = pd.read_csv(FEATURES_PATH)

    print("Loading user profiles...")
    user_profiles = load_user_profiles(USER_PROFILES_PATH)

    all_results = []

    for model_name, model_path in MODEL_PATHS.items():
        if not model_path.exists():
            print(f"Skipping {model_name}. Missing model file: {model_path}")
            continue

        print(f"\nEvaluating model: {model_name}")
        model = joblib.load(model_path)

        for index, user_profile in enumerate(user_profiles, start=1):
            result = evaluate_profile(
                model_name=model_name,
                model=model,
                restaurants_df=restaurants_df,
                user_profile=user_profile,
            )

            all_results.append(result)

            if index % 50 == 0:
                print(f"Evaluated {index} user profiles for {model_name}")

    results_df = pd.DataFrame(all_results)

    DETAIL_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(DETAIL_RESULTS_PATH, index=False)

    summary_df = (
        results_df.groupby("model")
        .agg(
            profiles_evaluated=("user_id", "count"),
            avg_candidates=("num_candidates", "mean"),
            avg_relevant=("num_relevant", "mean"),
            precision_at_5=("precision_at_5", "mean"),
            precision_at_10=("precision_at_10", "mean"),
            ndcg_at_5=("ndcg_at_5", "mean"),
            ndcg_at_10=("ndcg_at_10", "mean"),
            MAP=("average_precision", "mean"),
        )
        .reset_index()
    )

    summary_df.to_csv(SUMMARY_RESULTS_PATH, index=False)

    print("\nRanking Metric Summary:")
    print(summary_df)

    print(f"\nSaved detailed ranking metrics to: {DETAIL_RESULTS_PATH}")
    print(f"Saved ranking summary to: {SUMMARY_RESULTS_PATH}")


if __name__ == "__main__":
    main()
