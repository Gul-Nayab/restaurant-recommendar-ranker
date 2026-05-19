from src.pipeline.recommend import recommend


USER_PROFILES = [
    {
        "name": "Philadelphia Vegan Budget User",
        "city": "Philadelphia",
        "state": "PA",
        "latitude": 39.95,
        "longitude": -75.16,
        "max_distance": 10,
        "preferred_cuisines": ["vegan", "thai", "mexican"],
        "dietary_restrictions": ["vegan"],
        "price_range": [1, 2],
    },
    {
        "name": "Tampa Halal User",
        "city": "Tampa",
        "state": "FL",
        "latitude": 27.95,
        "longitude": -82.46,
        "max_distance": 10,
        "preferred_cuisines": ["middle eastern", "indian", "pakistani"],
        "dietary_restrictions": ["halal"],
        "price_range": [1, 2, 3],
    },
    {
        "name": "Tucson Gluten-Free Italian User",
        "city": "Tucson",
        "state": "AZ",
        "latitude": 32.22,
        "longitude": -110.97,
        "max_distance": 10,
        "preferred_cuisines": ["italian", "pizza"],
        "dietary_restrictions": ["gluten_free"],
        "price_range": [2, 3],
    },
    {
        "name": "Indianapolis Vegetarian Brunch User",
        "city": "Indianapolis",
        "state": "IN",
        "latitude": 39.77,
        "longitude": -86.16,
        "max_distance": 8,
        "preferred_cuisines": ["brunch", "breakfast", "cafe"],
        "dietary_restrictions": ["vegetarian"],
        "price_range": [1, 2],
    },
    {
        "name": "Nashville Budget Pizza User",
        "city": "Nashville",
        "state": "TN",
        "latitude": 36.16,
        "longitude": -86.78,
        "max_distance": 8,
        "preferred_cuisines": ["pizza", "italian"],
        "dietary_restrictions": [],
        "price_range": [1, 2],
    },
]


def print_recommendations(profile, model_name, results):
    print(f"Recommendations for: {profile['name']}")

    if results.empty:
        print("No restaurants found for this profile.")
        return

    for _, row in results.iterrows():
        print(f"\n{row['name']} ({row['city']}, {row['state']})")
        print(f"Stars: {row['stars']}")
        print(f"Review Count: {row['review_count']}")
        print(f"Price Level: {row['price_level']}")
        print(f"Distance: {row['distance']:.2f} miles")
        print(f"Score: {row['score']:.4f}")
        print(f"Categories: {row['categories']}")
        print(f"Explanation: {row['explanation']}")


def main():
    model_name = "xgboost"

    for profile in USER_PROFILES:
        results = recommend(
            user_profile=profile,
            model_name=model_name,
            top_n=5,
        )

        print_recommendations(profile, model_name, results)


if __name__ == "__main__":
    main()
