#!/usr/bin/env python3
import os
from collections import defaultdict


def count_series_per_study(data_dir):
    study_series_count = defaultdict(int)

    # Traverse through all the directories
    for study_id in os.listdir(data_dir):
        study_path = os.path.join(data_dir, study_id)
        if os.path.isdir(study_path):
            series_count = len([series for series in os.listdir(
                study_path) if os.path.isdir(os.path.join(study_path, series))])
            study_series_count[series_count] += 1

    return study_series_count


def print_series_count(data_dir):
    series_count_dict = count_series_per_study(data_dir)
    for series_count, num_studies in sorted(series_count_dict.items()):
        print(f"{series_count} series: {num_studies}")


if __name__ == "__main__":
    data_dir = "small_set/train_images"  # Change this to your data directory
    print_series_count(data_dir)
