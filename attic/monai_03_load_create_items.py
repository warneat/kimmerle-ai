import os
import pandas as pd
from monai.transforms import LoadImage
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
        label = row['condition']
        items.append({"image": img_path, "label": label})
    return items


# Create dataset items
root_dir = "/Users/dominikkimmerle/Documents/Studium/Master_01/ai-in-python-kimmerle/small_set/train_images"
logger.info("Creating dataset items.")
dataset_items = create_dataset_items(root_dir, label_annotations)

# Log the number of items
logger.info(f"Number of dataset items: {len(dataset_items)}")


# Validate that files exist and log paths
for item in dataset_items:
    if not os.path.isfile(item["image"]):
        logger.error(f"File does not exist: {item['image']}")
    else:
        logger.info(f"Valid file path: {item['image']}")

# Test loading a single image
test_image_path = dataset_items[0]["image"]
load_image_transform = LoadImage(image_only=True)

try:
    image = load_image_transform(test_image_path)
    logger.info("Single image loaded successfully.")
except Exception as e:
    logger.error(f"Error loading single image: {e}")

logger.info("Script execution complete.")
