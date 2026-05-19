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
        "name": "Gluten-Free Italian User",
        "latitude": 39.95,
        "longitude": -75.16,
        "max_distance": 10,
        "preferred_cuisines": ["italian", "pizza"],
        "dietary_restrictions": ["gluten_free"],
        "price_range": [2, 3],
    },
    {
        "name": "Vegetarian Brunch User",
        "latitude": 39.95,
        "longitude": -75.16,
        "max_distance": 6,
        "preferred_cuisines": ["brunch", "breakfast", "cafe"],
        "dietary_restrictions": ["vegetarian"],
        "price_range": [1, 2],
    },
    {
        "name": "Budget Pizza User",
        "latitude": 39.95,
        "longitude": -75.16,
        "max_distance": 5,
        "preferred_cuisines": ["pizza", "italian"],
        "dietary_restrictions": [],
        "price_range": [1, 2],
    },
    {
        "name": "High-End Sushi User",
        "latitude": 39.95,
        "longitude": -75.16,
        "max_distance": 12,
        "preferred_cuisines": ["sushi", "japanese"],
        "dietary_restrictions": [],
        "price_range": [3, 4],
    },
]


def print_recommendations(profile, model_name, results):
    print(f"Recommendations for: {profile['name']}")

    for _, row in results.iterrows():
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
        # for model_name in ["logistic", "xgboost"]:
        model_name = "xgboost"
        results = recommend(profile, model_name=model_name, top_n=5)
        print_recommendations(profile, model_name, results)


if __name__ == "__main__":
    main()
