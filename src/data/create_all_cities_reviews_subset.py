from pathlib import Path

from src.data.create_review_subset import create_review_subset


BUSINESS_SUBSET_PATH = Path("data/processed/restaurants_all_cities.csv")
OUTPUT_REVIEW_PATH = Path("data/processed/reviews_all_cities.csv")


def create_all_cities_reviews():
    create_review_subset(
        business_subset_path=BUSINESS_SUBSET_PATH,
        output_review_subset_path=OUTPUT_REVIEW_PATH,
        chunk_size=10000,
    )


if __name__ == "__main__":
    create_all_cities_reviews()
