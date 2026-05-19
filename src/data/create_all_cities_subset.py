from pathlib import Path
import pandas as pd

from src.data.create_city_subset import (
    load_business_data,
    filter_restaurants,
    select_useful_columns,
)


OUTPUT_PATH = Path("data/processed/restaurants_all_cities.csv")


def create_all_cities_subset() -> pd.DataFrame:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    business_df = load_business_data()
    print("Loaded business data:", business_df.shape)

    restaurant_df = filter_restaurants(business_df)
    print("Restaurant rows:", restaurant_df.shape)

    restaurant_df = select_useful_columns(restaurant_df)

    restaurant_df.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved all-cities restaurant dataset to: {OUTPUT_PATH}")
    print("Final shape:", restaurant_df.shape)

    print("\nTop city counts:")
    print(restaurant_df["city"].value_counts().head(20))

    return restaurant_df


if __name__ == "__main__":
    create_all_cities_subset()
