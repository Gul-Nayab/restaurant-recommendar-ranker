from pathlib import Path
import pandas as pd


business_data_subset_path = Path("data/processed/restaurants_philadelphia.csv")
raw_review_data_path = Path("data/raw/yelp_dataset/yelp_academic_dataset_review.json")
output_review_subset_path = Path("data/processed/reviews_philadelphia.csv")


business_df = pd.read_csv(business_data_subset_path)
business_ids = set(business_df["business_id"].unique())

keep_cols = ["review_id", "business_id", "stars", "text", "date", "useful"]
chunk_size = 10000
first_write = True

print("Loading reviews in chunks...")

for chunk in pd.read_json(raw_review_data_path, lines=True, chunksize=chunk_size):
    filtered_chunk = chunk[chunk["business_id"].isin(business_ids)]

    if not filtered_chunk.empty:
        existing_cols = [col for col in keep_cols if col in filtered_chunk.columns]
        filtered_chunk = filtered_chunk[existing_cols]

        filtered_chunk.to_csv(
            output_review_subset_path,
            mode="w" if first_write else "a",
            header=first_write,
            index=False,
        )
        first_write = False

print("Saved review subset:", output_review_subset_path)
