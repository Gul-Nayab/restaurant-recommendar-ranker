from pathlib import Path
import pandas as pd


def filter_restaurants(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["categories"] = df["categories"].fillna("")
    restaurant_df = df[
        df["categories"].str.contains("Restaurants", case=False, na=False)
    ]
    return restaurant_df


def filter_city(df: pd.DataFrame, city_name: str) -> pd.DataFrame:
    return df[df["city"].str.lower() == city_name.lower()].copy()


def select_useful_columns(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = [
        "business_id",
        "name",
        "city",
        "state",
        "latitude",
        "longitude",
        "stars",
        "review_count",
        "categories",
        "attributes",
        "is_open",
    ]
    existing_cols = [col for col in keep_cols if col in df.columns]
    return df[existing_cols].copy()


business_dataset_path = "data/raw/yelp_dataset/yelp_academic_dataset_business.json"
business_df = pd.read_json(business_dataset_path, lines=True)
print("Loaded business data:", business_df.shape)

restaurant_df = filter_restaurants(business_df)
print("Restaurant rows:", restaurant_df.shape)

# Choose one city to start with
city_df = filter_city(restaurant_df, "Philadelphia")
print("Philadelphia restaurants:", city_df.shape)

city_df = select_useful_columns(city_df)

output_path = "data/processed/restaurants_philadelphia.csv"
city_df.to_csv(output_path, index=False)

print(f"Saved subset to: {output_path}")
