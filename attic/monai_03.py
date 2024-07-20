import os
import pandas as pd
from monai.transforms import (
    LoadImage, EnsureChannelFirst, ScaleIntensity, EnsureType,
    Compose, Resize
)
from monai.data import Dataset, DataLoader
import logging

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load annotations
try:
    logger.info("Loading annotations.")
    label_annotations = pd.read_csv(
        "/Users/dominikkimmerle/Documents/Studium/Master_01/ai-in-python-kimmerle/cleaned_train_label_coordinates.csv")
except FileNotFoundError as e:
    logger.error(f"Error loading annotations: {e}")
    raise

# Function to create dataset items


def create_dataset_items(root_dir, annotations):
    items = []
    for _, row in annotations.iterrows():
        img_path = os.path.join(root_dir, str(row['study_id']), str(
            row['series_id']), f"{row['instance_number']}.dcm")
        if not os.path.exists(img_path):
            logger.warning(f"File not found: {img_path}")
            continue
        logger.info(f"Adding image: {img_path}")
        label = row[['x', 'y']].values.astype('float32')
        items.append({"image": img_path, "label": label})
    return items


# Create dataset items
root_dir = "/Users/dominikkimmerle/Documents/Studium/Master_01/ai-in-python-kimmerle/small_set/train_images"
logger.info("Creating dataset items.")
dataset_items = create_dataset_items(root_dir, label_annotations)

# Log the number of items
logger.info(f"Number of dataset items: {len(dataset_items)}")

# Validate that files exist
for item in dataset_items:
    if not os.path.isfile(item["image"]):
        logger.error(f"File does not exist: {item['image']}")

# Define transforms
logger.info("Defining transforms.")
transforms = Compose([
    LoadImage(image_only=True),
    EnsureChannelFirst(),
    ScaleIntensity(),
    Resize((224, 224)),
    EnsureType()
])

# Create MONAI dataset
logger.info("Creating MONAI dataset.")
dataset = Dataset(data=dataset_items, transform=transforms)

# Create DataLoader
logger.info("Creating DataLoader.")
data_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Iterate through the DataLoader to check if data loads correctly
try:
    for batch in data_loader:
        images, labels = batch["image"], batch["label"]
        logger.info(f"Loaded batch with {len(images)} images and labels.")
        # Just load one batch for now, break after first iteration
        break
except Exception as e:
    logger.error(f"Error during DataLoader iteration: {e}")

logger.info("Data loading complete.")
