import pandas as pd

from src.utils.geo import haversine


def cuisine_match(categories, preferred_cuisines):
    if pd.isna(categories):
        return 0

    categories = str(categories).lower()

    return int(any(cuisine.lower() in categories for cuisine in preferred_cuisines))


def get_dietary_column(dietary_restriction):
    mapping = {
        "vegan": "has_vegan_signal",
        "vegetarian": "has_vegetarian_signal",
        "halal": "has_halal_signal",
        "gluten_free": "has_gluten_free_signal",
        "gluten-free": "has_gluten_free_signal",
    }

    return mapping.get(dietary_restriction.lower())


def dietary_match(row, dietary_restrictions):
    if not dietary_restrictions:
        return 1

    matches = []

    for restriction in dietary_restrictions:
        col = get_dietary_column(restriction)

        if col and col in row:
            matches.append(int(row[col] == 1))
        else:
            matches.append(0)

    return int(all(matches))


def add_user_features(df, user_profile):
    df = df.copy()

    df["distance"] = haversine(
        user_profile["latitude"],
        user_profile["longitude"],
        df["latitude"],
        df["longitude"],
    )

    df["within_distance"] = (df["distance"] <= user_profile["max_distance"]).astype(int)

    df["cuisine_match"] = df["categories"].apply(
        lambda categories: cuisine_match(
            categories,
            user_profile.get("preferred_cuisines", []),
        )
    )

    df["price_match"] = df["price_level"].apply(
        lambda price: (
            int(price in user_profile.get("price_range", [])) if pd.notna(price) else 0
        )
    )

    df["dietary_match"] = df.apply(
        lambda row: dietary_match(
            row,
            user_profile.get("dietary_restrictions", []),
        ),
        axis=1,
    )

    return df
