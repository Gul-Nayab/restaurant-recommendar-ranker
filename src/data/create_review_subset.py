from pathlib import Path
import pandas as pd


RAW_REVIEW_PATH = Path("data/raw/yelp_dataset/yelp_academic_dataset_review.json")

KEEP_COLS = ["review_id", "business_id", "stars", "text", "date", "useful"]


def load_business_ids(business_subset_path: Path) -> set:
    business_df = pd.read_csv(business_subset_path)
    return set(business_df["business_id"].unique())


def filter_review_chunk(chunk: pd.DataFrame, business_ids: set) -> pd.DataFrame:
    filtered_chunk = chunk[chunk["business_id"].isin(business_ids)].copy()

    if filtered_chunk.empty:
        return filtered_chunk

    existing_cols = [col for col in KEEP_COLS if col in filtered_chunk.columns]
    return filtered_chunk[existing_cols]


def create_review_subset(
    business_subset_path: Path,
    output_review_subset_path: Path,
    raw_review_path: Path = RAW_REVIEW_PATH,
    chunk_size: int = 10000,
) -> None:
    output_review_subset_path.parent.mkdir(parents=True, exist_ok=True)

    if output_review_subset_path.exists():
        output_review_subset_path.unlink()

    business_ids = load_business_ids(business_subset_path)

    first_write = True
    total_rows = 0
    chunk_number = 0

    print("Loading reviews in chunks...")
    print("Number of businesses:", len(business_ids))

    for chunk in pd.read_json(raw_review_path, lines=True, chunksize=chunk_size):
        chunk_number += 1

        filtered_chunk = filter_review_chunk(chunk, business_ids)

        if not filtered_chunk.empty:
            filtered_chunk.to_csv(
                output_review_subset_path,
                mode="w" if first_write else "a",
                header=first_write,
                index=False,
            )

            first_write = False
            total_rows += len(filtered_chunk)

        if chunk_number % 50 == 0:
            print(f"Processed {chunk_number} chunks. Saved rows so far: {total_rows}")

    print("Saved review subset:", output_review_subset_path)
    print("Review rows saved:", total_rows)


if __name__ == "__main__":
    create_review_subset(
        business_subset_path=Path("data/processed/restaurants_all_cities.csv"),
        output_review_subset_path=Path("data/processed/reviews_all_cities.csv"),
    )
