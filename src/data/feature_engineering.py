from pathlib import Path
import re

import numpy as np
import pandas as pd


RESTAURANTS_PATH = Path("data/processed/restaurants_all_cities.csv")
REVIEWS_PATH = Path("data/processed/reviews_all_cities.csv")
OUTPUT_PATH = Path("data/processed/restaurant_features_all_cities.csv")


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


def category_contains(categories, keywords):
    if pd.isna(categories):
        return 0

    categories = str(categories).lower()
    return int(any(keyword in categories for keyword in keywords))


def add_basic_restaurant_features(restaurants_df):
    df = restaurants_df.copy()

    df["category_list"] = df["categories"].apply(clean_categories)
    df["category_count"] = df["category_list"].apply(len)

    df["price_level"] = df["attributes"].apply(extract_price)

    df["is_open"] = df["is_open"].fillna(0).astype(int)

    df["stars"] = df["stars"].fillna(df["stars"].median())
    df["review_count"] = df["review_count"].fillna(0)

    df["log_review_count"] = np.log1p(df["review_count"])

    # Category-based dietary signals
    df["category_vegan_signal"] = df["categories"].apply(
        lambda x: category_contains(x, ["vegan"])
    )
    df["category_vegetarian_signal"] = df["categories"].apply(
        lambda x: category_contains(x, ["vegetarian"])
    )
    df["category_halal_signal"] = df["categories"].apply(
        lambda x: category_contains(x, ["halal"])
    )
    df["category_gluten_free_signal"] = df["categories"].apply(
        lambda x: category_contains(x, ["gluten-free", "gluten free"])
    )

    return df


def build_review_features_from_csv(reviews_path: Path, chunksize: int = 50000):
    aggregate_chunks = []
    chunk_number = 0

    for chunk in pd.read_csv(reviews_path, chunksize=chunksize):
        chunk_number += 1

        chunk["text"] = chunk["text"].fillna("").astype(str).str.lower()
        chunk["review_length"] = chunk["text"].str.len()

        for feature_name, keywords in DIETARY_KEYWORDS.items():
            chunk[f"{feature_name}_review_mentions"] = chunk["text"].apply(
                lambda text: count_keyword_matches(text, keywords)
            )

        chunk["stars_sum"] = chunk["stars"]
        chunk["stars_count"] = chunk["stars"].notna().astype(int)
        chunk["review_text_count"] = 1
        chunk["review_length_sum"] = chunk["review_length"]
        chunk["useful_sum"] = chunk["useful"].fillna(0)

        grouped = (
            chunk.groupby("business_id")
            .agg(
                vegan_review_mentions=("vegan_review_mentions", "sum"),
                vegetarian_review_mentions=("vegetarian_review_mentions", "sum"),
                halal_review_mentions=("halal_review_mentions", "sum"),
                gluten_free_review_mentions=("gluten_free_review_mentions", "sum"),
                stars_sum=("stars_sum", "sum"),
                stars_count=("stars_count", "sum"),
                review_text_count=("review_text_count", "sum"),
                review_length_sum=("review_length_sum", "sum"),
                total_useful_votes=("useful_sum", "sum"),
            )
            .reset_index()
        )

        aggregate_chunks.append(grouped)

        if chunk_number % 25 == 0:
            print(f"Processed {chunk_number} review chunks...")

    if not aggregate_chunks:
        return pd.DataFrame(columns=["business_id"])

    all_review_features = pd.concat(aggregate_chunks, ignore_index=True)

    final = (
        all_review_features.groupby("business_id")
        .agg(
            vegan_review_mentions=("vegan_review_mentions", "sum"),
            vegetarian_review_mentions=("vegetarian_review_mentions", "sum"),
            halal_review_mentions=("halal_review_mentions", "sum"),
            gluten_free_review_mentions=("gluten_free_review_mentions", "sum"),
            stars_sum=("stars_sum", "sum"),
            stars_count=("stars_count", "sum"),
            review_text_count=("review_text_count", "sum"),
            review_length_sum=("review_length_sum", "sum"),
            total_useful_votes=("total_useful_votes", "sum"),
        )
        .reset_index()
    )

    final["avg_review_stars"] = final["stars_sum"] / final["stars_count"].replace(
        0, np.nan
    )
    final["avg_review_length"] = final["review_length_sum"] / final[
        "review_text_count"
    ].replace(0, np.nan)

    final = final.drop(columns=["stars_sum", "stars_count", "review_length_sum"])

    final["avg_review_stars"] = final["avg_review_stars"].fillna(0)
    final["avg_review_length"] = final["avg_review_length"].fillna(0)

    return final


def build_feature_table(restaurants_df, review_features_df):
    restaurant_features = add_basic_restaurant_features(restaurants_df)

    features_df = restaurant_features.merge(
        review_features_df,
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

    features_df["has_vegan_signal"] = (
        (features_df["vegan_review_mentions"] > 0)
        | (features_df["category_vegan_signal"] == 1)
    ).astype(int)

    features_df["has_vegetarian_signal"] = (
        (features_df["vegetarian_review_mentions"] > 0)
        | (features_df["category_vegetarian_signal"] == 1)
    ).astype(int)

    features_df["has_halal_signal"] = (
        (features_df["halal_review_mentions"] > 0)
        | (features_df["category_halal_signal"] == 1)
    ).astype(int)

    features_df["has_gluten_free_signal"] = (
        (features_df["gluten_free_review_mentions"] > 0)
        | (features_df["category_gluten_free_signal"] == 1)
    ).astype(int)

    return features_df


def create_feature_file(
    restaurants_path: Path = RESTAURANTS_PATH,
    reviews_path: Path = REVIEWS_PATH,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    if not restaurants_path.exists():
        raise FileNotFoundError(f"Missing restaurants file: {restaurants_path}")

    if not reviews_path.exists():
        raise FileNotFoundError(f"Missing reviews file: {reviews_path}")

    print(f"Loading restaurants from: {restaurants_path}")
    restaurants_df = pd.read_csv(restaurants_path)

    print(f"Building review features from: {reviews_path}")
    review_features_df = build_review_features_from_csv(reviews_path)

    print("Building final restaurant feature table...")
    features_df = build_feature_table(restaurants_df, review_features_df)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)

    print("Feature table saved to:", output_path)
    print("Shape:", features_df.shape)

    return features_df


def main():
    create_feature_file()


if __name__ == "__main__":
    main()
