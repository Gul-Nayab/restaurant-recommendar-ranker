from pathlib import Path
import json
import random

import pandas as pd


RESTAURANTS_PATH = Path("data/processed/restaurants_all_cities.csv")
OUTPUT_PATH = Path("data/processed/user_profiles_all_cities.json")


CUISINE_OPTIONS = [
    ["mexican", "vegan"],
    ["thai", "vegetarian"],
    ["indian", "halal"],
    ["middle eastern", "halal"],
    ["american", "gluten_free"],
    ["italian", "gluten_free"],
    ["pizza", "sandwiches"],
    ["chinese", "asian"],
    ["japanese", "sushi"],
    ["breakfast", "brunch"],
    ["mediterranean", "middle eastern"],
    ["burgers", "american"],
]


DIETARY_OPTIONS = [
    [],
    ["vegan"],
    ["vegetarian"],
    ["halal"],
    ["gluten_free"],
]


PRICE_OPTIONS = [
    [1],
    [1, 2],
    [2],
    [2, 3],
    [1, 2, 3],
    [3, 4],
]


def clean_name(value):
    return str(value).lower().replace(" ", "_").replace("/", "_")


def generate_profiles_for_city(city_name, state_name, city_df, profiles_per_city=3):
    profiles = []

    center_lat = city_df["latitude"].mean()
    center_lon = city_df["longitude"].mean()

    for i in range(profiles_per_city):
        profile = {
            "user_id": f"{clean_name(city_name)}_{clean_name(state_name)}_user_{i + 1}",
            "city": city_name,
            "state": state_name,
            "latitude": random.uniform(center_lat - 0.05, center_lat + 0.05),
            "longitude": random.uniform(center_lon - 0.05, center_lon + 0.05),
            "max_distance": random.choice([3, 5, 8, 10, 12]),
            "preferred_cuisines": random.choice(CUISINE_OPTIONS),
            "dietary_restrictions": random.choice(DIETARY_OPTIONS),
            "price_range": random.choice(PRICE_OPTIONS),
        }

        profiles.append(profile)

    return profiles


def generate_all_city_profiles(
    profiles_per_city=3,
    min_restaurants_per_city=10,
):
    restaurants_df = pd.read_csv(RESTAURANTS_PATH)

    restaurants_df = restaurants_df.dropna(
        subset=["city", "state", "latitude", "longitude"]
    )

    profiles = []

    grouped = restaurants_df.groupby(["city", "state"])

    for (city_name, state_name), city_df in grouped:
        if len(city_df) < min_restaurants_per_city:
            continue

        city_profiles = generate_profiles_for_city(
            city_name=city_name,
            state_name=state_name,
            city_df=city_df,
            profiles_per_city=profiles_per_city,
        )

        profiles.extend(city_profiles)

    return profiles


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    profiles = generate_all_city_profiles(
        profiles_per_city=3,
        min_restaurants_per_city=10,
    )

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(profiles, file, indent=4)

    print(f"Saved {len(profiles)} user profiles to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
