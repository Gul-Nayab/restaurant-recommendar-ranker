from pathlib import Path
import re
import numpy as np
import pandas as pd


RESTAURANTS_PATH = Path("../data/processed/restaurants_philadelphia.csv")
REVIEWS_PATH = Path("../data/processed/reviews_philadelphia.csv")
OUTPUT_PATH = Path("../data/processed/restaurant_features_philadelphia.csv")


DIETARY_KEYWORDS = {
    "vegan": ["vegan"],
    "vegetarian": ["vegetarian", "veggie"],
    "halal": ["halal"],
    "gluten_free": ["gluten free", "gluten-free", "glutenfree"],
}


def clean_categories(categories):
    if pd.isna(categories):
        return []
    return [cat.strip().lower() for cat in str(categories).split(",")]


def extract_price(attributes):
    """
    Yelp stores price in attributes as something like:
    {'RestaurantsPriceRange2': '2', ...}

    Since it may load as a string from CSV, this function uses regex.
    """
    if pd.isna(attributes):
        return np.nan

    attributes = str(attributes)

    match = re.search(r"RestaurantsPriceRange2['\"]?: ['\"]?(\d)", attributes)

    if match:
        return int(match.group(1))

    return np.nan


def count_keyword_matches(text, keywords):
    if pd.isna(text):
        return 0

    text = str(text).lower()
    return sum(text.count(keyword) for keyword in keywords)


def add_basic_restaurant_features(restaurants_df):
    df = restaurants_df.copy()

    df["category_list"] = df["categories"].apply(clean_categories)
    df["category_count"] = df["category_list"].apply(len)

    df["price_level"] = df["attributes"].apply(extract_price)

    df["is_open"] = df["is_open"].fillna(0).astype(int)

    df["stars"] = df["stars"].fillna(df["stars"].median())
    df["review_count"] = df["review_count"].fillna(0)

    # Log scale helps because review counts are usually very skewed
    df["log_review_count"] = np.log1p(df["review_count"])

    return df


def build_review_keyword_features(reviews_df):
    df = reviews_df.copy()
    df["text"] = df["text"].fillna("").str.lower()

    keyword_features = pd.DataFrame()
    keyword_features["business_id"] = df["business_id"]

    for feature_name, keywords in DIETARY_KEYWORDS.items():
        keyword_features[f"{feature_name}_review_mentions"] = df["text"].apply(
            lambda text: count_keyword_matches(text, keywords)
        )

    grouped = keyword_features.groupby("business_id").sum().reset_index()

    return grouped


def build_review_summary_features(reviews_df):
    df = reviews_df.copy()

    df["text"] = df["text"].fillna("")
    df["review_length"] = df["text"].str.len()

    summary = (
        df.groupby("business_id")
        .agg(
            avg_review_stars=("stars", "mean"),
            review_text_count=("text", "count"),
            avg_review_length=("review_length", "mean"),
            total_useful_votes=("useful", "sum"),
        )
        .reset_index()
    )

    return summary


def build_feature_table(restaurants_df, reviews_df):
    restaurant_features = add_basic_restaurant_features(restaurants_df)

    keyword_features = build_review_keyword_features(reviews_df)
    review_summary = build_review_summary_features(reviews_df)

    features_df = restaurant_features.merge(
        keyword_features,
        on="business_id",
        how="left",
    )

    features_df = features_df.merge(
        review_summary,
        on="business_id",
        how="left",
    )

    dietary_cols = [
        "vegan_review_mentions",
        "vegetarian_review_mentions",
        "halal_review_mentions",
        "gluten_free_review_mentions",
    ]

    for col in dietary_cols:
        features_df[col] = features_df[col].fillna(0)

    review_summary_cols = [
        "avg_review_stars",
        "review_text_count",
        "avg_review_length",
        "total_useful_votes",
    ]

    for col in review_summary_cols:
        features_df[col] = features_df[col].fillna(0)

    # Binary compatibility-style flags
    features_df["has_vegan_signal"] = (features_df["vegan_review_mentions"] > 0).astype(
        int
    )

    features_df["has_vegetarian_signal"] = (
        features_df["vegetarian_review_mentions"] > 0
    ).astype(int)

    features_df["has_halal_signal"] = (features_df["halal_review_mentions"] > 0).astype(
        int
    )

    features_df["has_gluten_free_signal"] = (
        features_df["gluten_free_review_mentions"] > 0
    ).astype(int)

    return features_df


def main():
    restaurants_df = pd.read_csv(RESTAURANTS_PATH)
    reviews_df = pd.read_csv(REVIEWS_PATH)

    features_df = build_feature_table(restaurants_df, reviews_df)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(OUTPUT_PATH, index=False)

    print("Feature table saved to:", OUTPUT_PATH)
    print("Shape:", features_df.shape)
    print("Columns:")
    print(features_df.columns.tolist())


if __name__ == "__main__":
    main()
