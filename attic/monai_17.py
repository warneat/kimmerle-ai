import os
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from monai.transforms import Compose, ScaleIntensity, EnsureType, Resize, EnsureChannelFirst
from monai.networks.nets import DenseNet121
from pydicom import dcmread
import numpy as np
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
csv_file = "./cleaned_train_label_coordinates.csv"
data_dir = "./small_set/train_images"
series_file = "./cleaned_train_series_descriptions.csv"

# Load CSV files
logger.info("Loading CSV files...")
data = pd.read_csv(csv_file)
series_descriptions = pd.read_csv(series_file)

# Merge series descriptions with data
data = data.merge(series_descriptions, on=["study_id", "series_id"])

# Filter necessary columns
data = data[["study_id", "series_id", "instance_number",
             "condition", "level", "series_description"]]

# Split the data into training and validation sets
train_data, val_data = train_test_split(
    data, test_size=0.2, random_state=42, stratify=data[['condition', 'level']])

# Define common transform for both axial and sagittal images
transforms_common = Compose([
    ScaleIntensity(),
    Resize((320, 320)),  # Resize all images to 320x320
    EnsureChannelFirst(),
    EnsureType(data_type="tensor")
])

# Create a custom dataset


class Spine3DDataset(Dataset):
    def __init__(self, data, data_dir, transform, num_slices=3):
        self.data = data
        self.data_dir = data_dir
        self.transform = transform
        self.num_slices = num_slices
        self.conditions = sorted(data['condition'].unique())
        self.levels = sorted(data['level'].unique())
        self.condition_to_index = {cond: idx for idx,
                                   cond in enumerate(self.conditions)}
        self.level_to_index = {lvl: idx for idx, lvl in enumerate(self.levels)}
        self.image_info = self._aggregate_image_info()

    def _aggregate_image_info(self):
        image_info = {}
        for idx, row in self.data.iterrows():
            image_id = (row['study_id'], row['series_id'],
                        row['instance_number'])
            if image_id not in image_info:
                image_info[image_id] = {'conditions': set(), 'levels': set(
                ), 'series_description': row['series_description']}
            image_info[image_id]['conditions'].add(
                self.condition_to_index[row['condition']])
            image_info[image_id]['levels'].add(
                self.level_to_index[row['level']])
        return image_info

    def __len__(self):
        return len(self.image_info)

    def __getitem__(self, idx):
        image_id = list(self.image_info.keys())[idx]
        study_id, series_id, instance_number = image_id
        slices = []

        for i in range(-self.num_slices // 2, self.num_slices // 2 + 1):
            slice_number = instance_number + i
            slice_path = os.path.join(self.data_dir, str(
                study_id), str(series_id), f'{slice_number}.dcm')
            if os.path.exists(slice_path):
                slice_image = dcmread(
                    slice_path).pixel_array.astype(np.float32)
                slices.append(slice_image)
            else:
                if len(slices) > 0:
                    # padding with zeros if slice does not exist
                    slices.append(np.zeros_like(slices[0]))
                else:
                    # initial zero slice if needed
                    slices.append(np.zeros((320, 320), dtype=np.float32))

        volume = np.stack(slices, axis=0)  # Stack slices to create a 3D volume
        volume = np.expand_dims(volume, axis=0)  # Add channel dimension
        volume = self.transform(volume)

        conditions = np.zeros(len(self.conditions), dtype=np.float32)
        levels = np.zeros(len(self.levels), dtype=np.float32)

        for cond in self.image_info[image_id]['conditions']:
            conditions[cond] = 1.0

        for lvl in self.image_info[image_id]['levels']:
            levels[lvl] = 1.0

        return {"image": volume, "conditions": conditions, "levels": levels}

# Define the model


class SpineStenosis3DClassifier(nn.Module):
    def __init__(self, num_conditions, num_levels):
        super(SpineStenosis3DClassifier, self).__init__()
        self.model = DenseNet121(
            spatial_dims=3, in_channels=1, out_channels=num_conditions + num_levels)

    def forward(self, x):
        return torch.split(self.model(x), [num_conditions, num_levels], dim=1)


# Create datasets and dataloaders for training and validation
logger.info("Creating datasets and dataloaders...")
train_dataset = Spine3DDataset(
    train_data, data_dir, transform=transforms_common)
val_dataset = Spine3DDataset(val_data, data_dir, transform=transforms_common)

# Adjust batch size based on memory availability and processing capability
batch_size = 4  # Adjust batch size based on memory for 3D CNNs

train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Training setup
num_conditions = len(train_dataset.conditions)
num_levels = len(train_dataset.levels)
model = SpineStenosis3DClassifier(num_conditions, num_levels)

criterion = nn.BCEWithLogitsLoss()
optimizer = Adam(model.parameters(), lr=5e-4)  # Adjusted learning rate
num_epochs = 80  # Increased number of epochs
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
logger.info("Starting training...")
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for i, batch_data in enumerate(train_dataloader):
        logger.info(f"Processing batch {i + 1}/{len(train_dataloader)}")
