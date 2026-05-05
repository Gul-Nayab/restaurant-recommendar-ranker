from src.pipeline.recommend import recommend


USER_PROFILES = [
    {
        "name": "Vegan Budget User",
        "latitude": 39.95,
        "longitude": -75.16,
        "max_distance": 10,
        "preferred_cuisines": ["vegan", "thai", "mexican"],
        "dietary_restrictions": ["vegan"],
        "price_range": [1, 2],
    },
    {
        "name": "Halal Food User",
        "latitude": 39.95,
        "longitude": -75.16,
        "max_distance": 8,
        "preferred_cuisines": ["middle eastern", "indian", "pakistani"],
        "dietary_restrictions": ["halal"],
        "price_range": [1, 2, 3],
    },
    {
        "name": "Gluten-Free User",
        "latitude": 39.95,
        "longitude": -75.16,
        "max_distance": 12,
        "preferred_cuisines": ["american", "italian", "brunch"],
        "dietary_restrictions": ["gluten_free"],
        "price_range": [2, 3],
    },
    {
        "name": "No Dietary Restriction User",
        "latitude": 39.95,
        "longitude": -75.16,
        "max_distance": 5,
        "preferred_cuisines": ["pizza", "burgers", "sandwiches"],
        "dietary_restrictions": [],
        "price_range": [1, 2],
    },
]


def print_recommendations(profile, results):
    print("\n" + "=" * 80)
    print(f"Recommendations for: {profile['name']}")
    print("=" * 80)

    for index, row in results.iterrows():
        print(f"\n{row['name']}")
        print(f"Stars: {row['stars']}")
        print(f"Review Count: {row['review_count']}")
        print(f"Price Level: {row['price_level']}")
        print(f"Distance: {row['distance']:.2f} miles")
        print(f"Score: {row['score']:.4f}")
        print(f"Categories: {row['categories']}")
        print(f"Explanation: {row['explanation']}")


def main():
    for profile in USER_PROFILES:
        results = recommend(profile, top_n=5)
        print_recommendations(profile, results)


if __name__ == "__main__":
    main()
