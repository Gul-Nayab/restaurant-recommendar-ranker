from pathlib import Path
import json
import random


OUTPUT_PATH = Path("data/processed/user_profiles_philadelphia.json")


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
]


def generate_profiles(n=50):
    profiles = []

    for i in range(n):
        profile = {
            "user_id": f"user_{i + 1}",
            "latitude": random.uniform(39.90, 40.05),
            "longitude": random.uniform(-75.25, -75.10),
            "max_distance": random.choice([3, 5, 8, 10, 12]),
            "preferred_cuisines": random.choice(CUISINE_OPTIONS),
            "dietary_restrictions": random.choice(DIETARY_OPTIONS),
            "price_range": random.choice(PRICE_OPTIONS),
        }

        profiles.append(profile)

    return profiles


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    profiles = generate_profiles(50)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as file:
        json.dump(profiles, file, indent=4)

    print(f"Saved {len(profiles)} user profiles to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
