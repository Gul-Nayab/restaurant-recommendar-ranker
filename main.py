from pathlib import Path

import pandas as pd

from src.pipeline.recommend import recommend


FEATURES_PATH = Path("data/processed/restaurant_features_all_cities.csv")

"""
Top available cities:

- Philadelphia, PA (5852 restaurants)
- Tampa, FL (2960 restaurants)
- Indianapolis, IN (2862 restaurants)
- Nashville, TN (2502 restaurants)
- Tucson, AZ (2466 restaurants)
- New Orleans, LA (2259 restaurants)
- Edmonton, AB (2166 restaurants)
- Saint Louis, MO (1790 restaurants)
- Reno, NV (1286 restaurants)
- Boise, ID (847 restaurants)
- Santa Barbara, CA (767 restaurants)
- Clearwater, FL (678 restaurants)
- Wilmington, DE (619 restaurants)
- St. Louis, MO (542 restaurants)
- Metairie, LA (522 restaurants)
- Saint Petersburg, FL (491 restaurants)
- Franklin, TN (442 restaurants)
- St. Petersburg, FL (404 restaurants)
- Sparks, NV (334 restaurants)
- Brandon, FL (326 restaurants)
"""


def load_city_data():
    df = pd.read_csv(FEATURES_PATH)

    city_data = (
        df.dropna(subset=["city", "state", "latitude", "longitude"])
        .groupby(["city", "state"])
        .agg(
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            restaurant_count=("business_id", "count"),
        )
        .reset_index()
    )

    return city_data


CITY_DATA = load_city_data()


def find_city(city_input):
    city_input = city_input.strip().lower()

    matches = CITY_DATA[CITY_DATA["city"].str.lower() == city_input]

    if matches.empty:
        return None

    matches = matches.sort_values(
        "restaurant_count",
        ascending=False,
    )

    return matches.iloc[0]


def choose_city():
    print("\nEnter any city from the Yelp dataset.")

    while True:
        city_input = input("\nCity name [default: Philadelphia]: ").strip()

        if not city_input:
            city_input = "Philadelphia"

        city_match = find_city(city_input)

        if city_match is None:
            print("City not found in dataset. Try another city.")
            continue

        print(
            f"\nUsing city: {city_match['city']}, "
            f"{city_match['state']} "
            f"({int(city_match['restaurant_count'])} restaurants)"
        )

        return {
            "city": city_match["city"],
            "state": city_match["state"],
            "lat": city_match["latitude"],
            "lon": city_match["longitude"],
        }


def get_list_input(prompt):
    value = input(prompt).strip()

    if not value:
        return []

    return [item.strip().lower() for item in value.split(",") if item.strip()]


def get_price_range():
    value = input("Price range, comma-separated [default: 1,2,3]: ").strip()

    if not value:
        return [1, 2, 3]

    prices = []

    for item in value.split(","):
        item = item.strip()

        if item.isdigit():
            prices.append(int(item))

    return prices if prices else [1, 2, 3]


def get_float_input(prompt, default):
    value = input(f"{prompt} [default: {default:.4f}]: ").strip()

    if not value:
        return float(default)

    return float(value)


def get_int_input(prompt, default):
    value = input(f"{prompt} [default: {default}]: ").strip()

    if not value:
        return default

    return int(value)


def build_user_profile():
    print("Personalized Restaurant Recommendation System\n")

    city_data = choose_city()

    latitude = get_float_input("Latitude", city_data["lat"])
    longitude = get_float_input("Longitude", city_data["lon"])
    max_distance = get_float_input("Max travel distance in miles", 10)

    preferred_cuisines = get_list_input(
        "Preferred cuisines, comma-separated ex: vegan, thai, mexican: "
    )

    dietary_restrictions = get_list_input(
        "Dietary restrictions, comma-separated ex: vegan, halal, gluten_free: "
    )

    price_range = get_price_range()

    return {
        "name": f"{city_data['city']} User",
        "city": city_data["city"],
        "state": city_data["state"],
        "latitude": latitude,
        "longitude": longitude,
        "max_distance": max_distance,
        "preferred_cuisines": preferred_cuisines,
        "dietary_restrictions": dietary_restrictions,
        "price_range": price_range,
    }


def print_recommendations(profile, results):
    print(f"Top Recommendations for {profile['city']}, {profile['state']}")

    if results.empty:
        print("No restaurants found. Try increasing max distance.")
        return

    for rank, (_, row) in enumerate(results.iterrows(), start=1):
        print(f"\n#{rank}: {row['name']} ({row['city']}, {row['state']})")
        print(f"Rating: {row['stars']} stars | Reviews: {int(row['review_count'])}")
        print(f"Price Level: {row['price_level']}")
        print(f"Distance: {row['distance']:.2f} miles")
        print(f"Model Score: {row['score']:.4f}")
        print(f"Categories: {row['categories']}")
        print(f"Why: {row['explanation']}")


def main():
    while True:
        profile = build_user_profile()

        top_n = get_int_input("How many recommendations?", 5)

        results = recommend(
            user_profile=profile,
            model_name="xgboost",
            top_n=top_n,
        )

        print_recommendations(profile, results)

        again = input("\nSearch again? (y/n): ").strip().lower()

        if again != "y":
            break


if __name__ == "__main__":
    main()
