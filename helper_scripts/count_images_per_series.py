#!/usr/bin/env python3
import os
from collections import defaultdict

# Define the path to the data directory
data_dir = "small_set/train_images"

# Dictionary to hold counts of images per series
series_image_counts = defaultdict(int)

# Walk through the directories
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.endswith('.dcm'):
            path_parts = root.split(os.sep)
            if len(path_parts) >= 2:
                study_id = path_parts[-2]
                series_id = path_parts[-1]
                series_image_counts[(study_id, series_id)] += 1

# Create a dictionary to hold counts and their corresponding series counts
count_series_dict = defaultdict(int)
for series, count in series_image_counts.items():
    count_series_dict[count] += 1

# Sort the dictionary by the number of images
sorted_counts = sorted(count_series_dict.items())

# Print the sorted counts
for count, num_series in sorted_counts:
    print(f"{count}: {num_series} series")
