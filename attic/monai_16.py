import os
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
from monai.transforms import Compose, ScaleIntensity, EnsureType, Resize
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
    EnsureType(data_type="tensor")
])

# Create a custom dataset


class SpineDataset(Dataset):
    def __init__(self, data, data_dir, transform):
        self.data = data
        self.data_dir = data_dir
        self.transform = transform
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
        image_path = os.path.join(self.data_dir, str(
            study_id), str(series_id), f'{instance_number}.dcm')
        image = dcmread(image_path).pixel_array.astype(np.float32)
        image = np.expand_dims(image, axis=0)

        image = self.transform(image)

        conditions = np.zeros(len(self.conditions), dtype=np.float32)
        levels = np.zeros(len(self.levels), dtype=np.float32)

        for cond in self.image_info[image_id]['conditions']:
            conditions[cond] = 1.0

        for lvl in self.image_info[image_id]['levels']:
            levels[lvl] = 1.0

        return {"image": image, "conditions": conditions, "levels": levels}


# Create datasets and dataloaders for training and validation
logger.info("Creating datasets and dataloaders...")
train_dataset = SpineDataset(train_data, data_dir, transform=transforms_common)
val_dataset = SpineDataset(val_data, data_dir, transform=transforms_common)

# Adjust batch size based on memory availability and processing capability
batch_size = 64  # Adjust batch size based on memory

train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define the model


class SpineStenosisClassifier(nn.Module):
    def __init__(self, num_conditions, num_levels):
        super(SpineStenosisClassifier, self).__init__()
        self.model = DenseNet121(
            spatial_dims=2, in_channels=1, out_channels=num_conditions + num_levels)

    def forward(self, x):
        return torch.split(self.model(x), [num_conditions, num_levels], dim=1)


num_conditions = len(train_dataset.conditions)
num_levels = len(train_dataset.levels)
model = SpineStenosisClassifier(num_conditions, num_levels)

# Training setup
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
        inputs = batch_data["image"].to(device)
        condition_labels = batch_data["conditions"].to(device)
        level_labels = batch_data["levels"].to(device)

        optimizer.zero_grad()
        condition_logits, level_logits = model(inputs)
        condition_loss = criterion(condition_logits, condition_labels)
        level_loss = criterion(level_logits, level_labels)
        loss = condition_loss + level_loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    logger.info(
        f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_dataloader)}")

# Validation
logger.info("Starting validation...")
model.eval()
y_true_conditions = []
y_pred_conditions = []
y_true_levels = []
y_pred_levels = []

with torch.no_grad():
    for i, batch_data in enumerate(val_dataloader):
        logger.info(f"Validating batch {i + 1}/{len(val_dataloader)}")
        inputs = batch_data["image"].to(device)
        condition_labels = batch_data["conditions"].to(device)
        level_labels = batch_data["levels"].to(device)

        condition_logits, level_logits = model(inputs)
        condition_preds = torch.sigmoid(condition_logits) > 0.5
        level_preds = torch.sigmoid(level_logits) > 0.5

        y_true_conditions.extend(condition_labels.cpu().numpy())
        y_pred_conditions.extend(condition_preds.cpu().numpy())
        y_true_levels.extend(level_labels.cpu().numpy())
        y_pred_levels.extend(level_preds.cpu().numpy())

# Evaluate results
logger.info("Condition Classification Report:")
logger.info("\n" + classification_report(np.array(y_true_conditions),
            np.array(y_pred_conditions), target_names=train_dataset.conditions))

logger.info("Level Classification Report:")
logger.info("\n" + classification_report(np.array(y_true_levels),
            np.array(y_pred_levels), target_names=train_dataset.levels))
