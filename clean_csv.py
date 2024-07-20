#!/usr/bin/env python3
import pandas as pd
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths
input_dir = './original_csv'
train_images_dir = './small_set/train_images'

# Load the CSV files
logging.info("Loading CSV files...")
train_series_description = pd.read_csv(
    os.path.join(input_dir, 'train_series_descriptions.csv'))
train_label_coordinates = pd.read_csv(
    os.path.join(input_dir, 'train_label_coordinates.csv'))

# Get the list of available study directories
logging.info("Getting list of available study directories...")
available_studies = {study_id for study_id in os.listdir(
    train_images_dir) if os.path.isdir(os.path.join(train_images_dir, study_id))}
logging.info(f"Found {len(available_studies)} available studies.")

# Filter the train_series_description to only include available studies
logging.info(
    "Filtering train_series_description to include only available studies...")
train_series_description = train_series_description[train_series_description['study_id'].astype(
    str).isin(available_studies)]
logging.info(
    f"Filtered train_series_description to {len(train_series_description)} rows.")

# Remove unavailable series_ids from the filtered train_series_description
logging.info("Removing unavailable series_ids from train_series_description...")
valid_series_ids = {
    series_id
    for study_id in available_studies
    for series_id in os.listdir(os.path.join(train_images_dir, study_id))
    if os.path.isdir(os.path.join(train_images_dir, study_id, series_id))
}
logging.info(f"Found {len(valid_series_ids)} valid series IDs.")
train_series_description = train_series_description[train_series_description['series_id'].astype(
    str).isin(valid_series_ids)]
logging.info(
    f"Filtered train_series_description to {len(train_series_description)} rows with valid series IDs.")

# Function to process series descriptions and label coordinates


def process_series(series_description_df, label_coordinates_df, series_type, conditions, levels):
    logging.info(f"Processing {series_type} series...")
    series_ids = series_description_df[series_description_df['series_description'].str.contains(
        series_type, na=False)]['series_id'].astype(str)

    if series_ids.empty:
        logging.warning(f"No series found for {series_type}.")
        return series_description_df, label_coordinates_df

    filtered_series_description = series_description_df[~series_description_df['series_description'].str.contains(
        series_type, na=False)]
    series_description = series_description_df[series_description_df['series_id'].astype(
        str).isin(series_ids)]

    filtered_label_coordinates = label_coordinates_df[~label_coordinates_df['series_id'].astype(
        str).isin(series_ids)]
    series_label_coordinates = label_coordinates_df[label_coordinates_df['series_id'].astype(
        str).isin(series_ids)]

    if series_label_coordinates.empty:
        logging.warning(f"No label coordinates found for {series_type}.")
        return filtered_series_description, filtered_label_coordinates

    logging.info(f"Saving {series_type.lower()} series description to CSV...")
    series_description.to_csv(
        f'train_series_description_{series_type.lower()}.csv', index=False)
    logging.info(f"Saving {series_type.lower()} label coordinates to CSV...")
    series_label_coordinates.to_csv(
        f'train_label_coordinates_{series_type.lower()}.csv', index=False)

    # Create the filepath column and aggregate conditions and levels for each unique filepath
    def create_aggregated_dataframe(df, conditions, levels):
        df = df.copy()
        df['filepath'] = df.apply(
            lambda row: f"./small_set/train_images/{row['study_id']}/{row['series_id']}/{row['instance_number']}.dcm", axis=1)
        aggregated_df = df.groupby('filepath').agg({
            'study_id': 'first',
            'series_id': 'first',
            'instance_number': 'first',
            'condition': lambda x: ', '.join(sorted(set(x))),
            'level': lambda x: ', '.join(sorted(set(x)))
        }).reset_index()

        for condition in conditions:
            aggregated_df[condition] = aggregated_df['condition'].apply(
                lambda x: condition in x)
        for level in levels:
            aggregated_df[level] = aggregated_df['level'].apply(
                lambda x: level in x)

        aggregated_df.drop(columns=['condition', 'level'], inplace=True)
        columns = ['filepath', 'study_id', 'series_id',
                   'instance_number'] + conditions + levels
        return aggregated_df[columns]

    logging.info(
        f"Creating aggregated dataframe for {series_type.lower()} series...")
    aggregated_df = create_aggregated_dataframe(
        series_label_coordinates, conditions, levels)
    logging.info(
        f"Saving aggregated {series_type.lower()} label coordinates to CSV with file paths...")
    aggregated_df.to_csv(
        f'train_label_coordinates_{series_type.lower()}_paths.csv', index=False)

    # Verify file paths
    verify_file_paths(
        aggregated_df, f'train_label_coordinates_{series_type.lower()}_paths.csv')

    return filtered_series_description, filtered_label_coordinates


def verify_file_paths(df, filename):
    missing_files = []
    for path in df['filepath']:
        if not os.path.exists(path):
            missing_files.append(path)
    if missing_files:
        logging.warning(f"Missing {len(missing_files)} files in {filename}:")
        for path in missing_files:
            logging.warning(path)
    else:
        logging.info(f"All file paths in {filename} are valid.")


# Axial and sagittal conditions and levels
axial_conditions = ['Left Subarticular Stenosis',
                    'Right Subarticular Stenosis']
axial_levels = ['L1/L2', 'L2/L3', 'L3/L4', 'L4/L5', 'L5/S1']
sagittal_conditions = ['Spinal Canal Stenosis',
                       'Right Neural Foraminal Narrowing', 'Left Neural Foraminal Narrowing']
sagittal_levels = axial_levels

# Process axial and sagittal series
logging.info("Processing axial series...")
train_series_description, filtered_train_label_coordinates = process_series(
    train_series_description, train_label_coordinates, 'Axial', axial_conditions, axial_levels)
logging.info("Processing sagittal series...")
train_series_description, filtered_train_label_coordinates = process_series(
    train_series_description, filtered_train_label_coordinates, 'Sagittal', sagittal_conditions, sagittal_levels)

# Traverse through the available study directories and log invalid series_id directories
logging.info("Checking for invalid series_id directories...")
for study_id in available_studies:
    study_path = os.path.join(train_images_dir, study_id)
    for series_id in os.listdir(study_path):
        series_path = os.path.join(study_path, series_id)
        if os.path.isdir(series_path) and series_id not in valid_series_ids:
            logging.info(f"Invalid series directory found: {series_path}")

logging.info("Script execution completed.")
