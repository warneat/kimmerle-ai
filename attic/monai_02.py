'''First steps with bare minimum like loading data...'''

import os
import pandas as pd

# import monai-specific
from monai.transforms import LoadImage, EnsureChannelFirst, ScaleIntensity, EnsureType, Compose, Resize
from monai.data import Dataset, DataLoader

#logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load annotations
try:
    logger.info("Loading annotations.")
    label_annotations = pd.read_csv(
        "/Users/dominikkimmerle/Documents/Studium/Master_01/ai-in-python-kimmerle/train_label_coordinates.csv")
except FileNotFoundError as e:
    logger.error(f"Error loading annotations: {e}")
    raise

#TODO:
# create dataset items (not losading the actual image, only path and label)

# Image loading and preprocessing