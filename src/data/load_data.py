import pandas as pd

business_dataset_path = "data/raw/yelp_dataset/yelp_academic_dataset_business.json"
business_df = pd.read_json(business_dataset_path, lines=True)

print("Business data loaded.")
print("Shape:", business_df.shape)
print("\nColumns:")
print(business_df.columns.tolist())
print("\nSample rows:")
print(business_df.head())
