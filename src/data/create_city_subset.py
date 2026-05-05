from pathlib import Path
import pandas as pd


RAW_BUSINESS_PATH = Path("data/raw/yelp_dataset/yelp_academic_dataset_business.json")
OUTPUT_DIR = Path("data/processed")


KEEP_COLS = [
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


def load_business_data(path: Path = RAW_BUSINESS_PATH) -> pd.DataFrame:
    return pd.read_json(path, lines=True)


def filter_restaurants(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["categories"] = df["categories"].fillna("")

    return df[df["categories"].str.contains("Restaurants", case=False, na=False)].copy()


def filter_city(df: pd.DataFrame, city_name: str) -> pd.DataFrame:
    return df[df["city"].str.lower() == city_name.lower()].copy()


def select_useful_columns(df: pd.DataFrame) -> pd.DataFrame:
    existing_cols = [col for col in KEEP_COLS if col in df.columns]
    return df[existing_cols].copy()


def make_output_path(city_name: str) -> Path:
    clean_city = city_name.lower().replace(" ", "_")
    return OUTPUT_DIR / f"restaurants_{clean_city}.csv"


def create_city_subset(city_name: str) -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    business_df = load_business_data()
    print("Loaded business data:", business_df.shape)

    restaurant_df = filter_restaurants(business_df)
    print("Restaurant rows:", restaurant_df.shape)

    city_df = filter_city(restaurant_df, city_name)
    print(f"{city_name} restaurants:", city_df.shape)

    city_df = select_useful_columns(city_df)

    output_path = make_output_path(city_name)
    city_df.to_csv(output_path, index=False)

    print(f"Saved subset to: {output_path}")

    return city_df


if __name__ == "__main__":
    create_city_subset("Philadelphia")
