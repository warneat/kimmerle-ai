#!/usr/bin/env python3
import pandas as pd

# Load the descriptions to identify sagittal images
series_descriptions = pd.read_csv('cleaned_train_series_descriptions.csv')

# Get the series_ids of sagittal images
sagittal_series_ids = series_descriptions[series_descriptions['series_description'].str.contains(
    "Sagittal")]['series_id'].tolist()

# Load the combined paths file
combined_paths = pd.read_csv('combined_paths.csv')

# Filter out rows where series_id is in sagittal_series_ids
filtered_paths = combined_paths[~combined_paths['image_path'].str.contains(
    '|'.join([str(sid) for sid in sagittal_series_ids]))]

# Save the result to axial_labels.csv
filtered_paths.to_csv('axial_image_labels.csv', index=False)

print("Filtered data saved to axial_labels.csv.")
