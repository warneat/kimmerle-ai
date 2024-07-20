#!/usr/bin/env python3
import os
import numpy as np
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from torch.optim import Adam
from monai.transforms import Compose, ScaleIntensity, Resize, EnsureType
from monai.networks.nets import DenseNet121
from pydicom import dcmread
import logging
import time
import matplotlib.pyplot as plt

# Ensure the log directory exists
log_dir = './logs'
os.makedirs(log_dir, exist_ok=True)
log_file_path = os.path.join(log_dir, 'training.log')

# Set up logging
logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger()

# Load the first dataset
data1 = pd.read_csv('train_label_coordinates_axial_paths.csv')
data1 = data1.set_index(
    ['filepath', 'study_id', 'series_id', 'instance_number'])
data1 = data1.replace({True: 1, False: 0})  # Convert True/False to 1/0

# Load the second dataset
data2 = pd.read_csv('train_label_coordinates_sagittal_paths.csv')
data2 = data2.set_index(
    ['filepath', 'study_id', 'series_id', 'instance_number'])
data2 = data2.replace({True: 1, False: 0})  # Convert True/False to 1/0

# Split datasets
train_data1, val_data1 = train_test_split(
    data1, test_size=0.2, random_state=42)
train_data2, val_data2 = train_test_split(
    data2, test_size=0.2, random_state=42)

# Number of possible outputs
num_conditions1 = 2  # 2 types of subarticular stenosis
num_levels1 = 5  # 5 levels
num_conditions2 = 3  # 1 type of canal stenosis + 2 foraminal narrowing
num_levels2 = 5  # 5 levels

# Define a custom dataset class for the first dataset


class SpineDataset1(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row.name[0]
        try:
            image = dcmread(image_path).pixel_array.astype(np.float32)
        except FileNotFoundError:
            logger.warning(f"File not found: {image_path}. Skipping.")
            return None  # Return None to indicate this sample should be skipped
        image = np.expand_dims(image, axis=0)

        if self.transform:
            image = self.transform(image)

        condition_labels = row[:num_conditions1].astype(np.float32).values
        level_labels = row[num_conditions1:].astype(np.float32).values

        return {"image": image, "condition": torch.tensor(condition_labels), "level": torch.tensor(level_labels)}

# Define a custom dataset class for the second dataset


class SpineDataset2(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row.name[0]
        try:
            image = dcmread(image_path).pixel_array.astype(np.float32)
        except FileNotFoundError:
            logger.warning(f"File not found: {image_path}. Skipping.")
            return None  # Return None to indicate this sample should be skipped
        image = np.expand_dims(image, axis=0)

        if self.transform:
            image = self.transform(image)

        condition_labels = row[:num_conditions2].astype(np.float32).values
        level_labels = row[num_conditions2:].astype(np.float32).values

        return {"image": image, "condition": torch.tensor(condition_labels), "level": torch.tensor(level_labels)}


# Define the transforms
transforms = Compose([
    ScaleIntensity(),
    Resize((320, 320)),
    EnsureType(data_type="tensor")
])

# Custom collate function to filter out None samples


def collate_fn(batch):
    # Filter out None samples
    batch = [item for item in batch if item is not None]
    if len(batch) == 0:
        return None  # Return None if the batch is empty
    return default_collate(batch)


# Create datasets and dataloaders for the first dataset
train_dataset1 = SpineDataset1(train_data1, transform=transforms)
val_dataset1 = SpineDataset1(val_data1, transform=transforms)
batch_size = 32
train_dataloader1 = DataLoader(
    train_dataset1, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataloader1 = DataLoader(
    val_dataset1, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Create datasets and dataloaders for the second dataset
train_dataset2 = SpineDataset2(train_data2, transform=transforms)
val_dataset2 = SpineDataset2(val_data2, transform=transforms)
train_dataloader2 = DataLoader(
    train_dataset2, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataloader2 = DataLoader(
    val_dataset2, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

# Define the first model


class SpineStenosisClassifier(nn.Module):
    def __init__(self, num_conditions, num_levels):
        super(SpineStenosisClassifier, self).__init__()
        self.backbone = DenseNet121(
            spatial_dims=2, in_channels=1, out_channels=512)
        self.condition_head = nn.Linear(512, num_conditions)
        self.level_head = nn.Linear(512, num_levels)

    def forward(self, x):
        features = self.backbone(x)
        condition_logits = self.condition_head(features)
        level_logits = self.level_head(features)
        return condition_logits, level_logits


model1 = SpineStenosisClassifier(num_conditions1, num_levels1)

# Define the second model


class SecondSpineClassifier(nn.Module):
    def __init__(self, num_conditions, num_levels):
        super(SecondSpineClassifier, self).__init__()
        self.backbone = DenseNet121(
            spatial_dims=2, in_channels=1, out_channels=512)
        self.condition_head = nn.Linear(512, num_conditions)
        self.level_head = nn.Linear(512, num_levels)

    def forward(self, x):
        features = self.backbone(x)
        condition_logits = self.condition_head(features)
        level_logits = self.level_head(features)
        return condition_logits, level_logits


model2 = SecondSpineClassifier(num_conditions2, num_levels2)

# Training setup
criterion = nn.BCEWithLogitsLoss()
optimizer1 = Adam(model1.parameters(), lr=5e-4)
optimizer2 = Adam(model2.parameters(), lr=5e-4)
num_epochs = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")
model1.to(device)
model2.to(device)

# Ensure the output directory exists
output_dir = './saved_models_and_plots'
os.makedirs(output_dir, exist_ok=True)
checkpoint_path1 = os.path.join(output_dir, 'checkpoint1.pt')
checkpoint_path2 = os.path.join(output_dir, 'checkpoint2.pt')

# Early stopping class


class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, val_loss, model, checkpoint_path):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, checkpoint_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            logger.info(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, checkpoint_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, checkpoint_path):
        '''Saves model when validation loss decreases.'''
        torch.save(model.state_dict(), checkpoint_path)
        logger.info(
            f'Validation loss decreased ({self.best_score:.6f} --> {val_loss:.6f}). Saving model to {checkpoint_path}')


# Early stopping parameters
# changed, since might have stopped to early ?
patience = 80
early_stopping1 = EarlyStopping(patience=patience, delta=0)
early_stopping2 = EarlyStopping(patience=patience, delta=0)

# Initialize lists to store metrics for the first model
train_losses1 = []
val_losses1 = []
train_accuracies1 = []
val_accuracies1 = []

# Training loop for the first model
logger.info("Starting training for the first model...")
for epoch in range(num_epochs):
    start_time = time.time()
    model1.train()
    epoch_loss = 0
    correct_train_condition = 0
    correct_train_level = 0
    total_train = 0

    for i, batch_data in enumerate(train_dataloader1):
        if batch_data is None:
            continue  # Skip empty batches
        inputs = batch_data["image"].to(device)
        condition_labels = batch_data["condition"].to(device)
        level_labels = batch_data["level"].to(device)

        optimizer1.zero_grad()
        condition_logits, level_logits = model1(inputs)
        condition_loss = criterion(condition_logits, condition_labels)
        level_loss = criterion(level_logits, level_labels)
        loss = condition_loss + level_loss
        loss.backward()
        optimizer1.step()
        epoch_loss += loss.item()

        condition_preds = torch.sigmoid(condition_logits) > 0.4
        level_preds = torch.sigmoid(level_logits) > 0.4
        correct_train_condition += (condition_preds ==
                                    condition_labels).sum().item()
        correct_train_level += (level_preds == level_labels).sum().item()
        total_train += condition_labels.numel()

    train_loss = epoch_loss / len(train_dataloader1)
    train_accuracy = (correct_train_condition +
                      correct_train_level) / (2 * total_train)
    train_losses1.append(train_loss)
    train_accuracies1.append(train_accuracy)

    logger.info(
        f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss}, Accuracy: {train_accuracy}")

    # Validation
    model1.eval()
    val_loss = 0
    correct_val_condition = 0
    correct_val_level = 0
    total_val = 0
    with torch.no_grad():
        for i, batch_data in enumerate(val_dataloader1):
            if batch_data is None:
                continue  # Skip empty batches
            inputs = batch_data["image"].to(device)
            condition_labels = batch_data["condition"].to(device)
            level_labels = batch_data["level"].to(device)

            condition_logits, level_logits = model1(inputs)
            condition_loss = criterion(condition_logits, condition_labels)
            level_loss = criterion(level_logits, level_labels)
            loss = condition_loss + level_loss
            val_loss += loss.item()

            condition_preds = torch.sigmoid(condition_logits) > 0.4
            level_preds = torch.sigmoid(level_logits) > 0.4

            correct_val_condition += (condition_preds ==
                                      condition_labels).sum().item()
            correct_val_level += (level_preds == level_labels).sum().item()
            total_val += condition_labels.numel()

    val_loss /= len(val_dataloader1)
    val_accuracy = (correct_val_condition +
                    correct_val_level) / (2 * total_val)
    val_losses1.append(val_loss)
    val_accuracies1.append(val_accuracy)

    logger.info(f"Validation Loss: {val_loss}, Accuracy: {val_accuracy}")

    early_stopping1(val_loss, model1, checkpoint_path1)
    if early_stopping1.early_stop:
        logger.info("Early stopping")
        break

# Load the best model for the first dataset
model1.load_state_dict(torch.load(checkpoint_path1))

# Final evaluation for the first model
logger.info("Starting final evaluation for the first model...")
model1.eval()
y_true_conditions1 = []
y_pred_conditions1 = []
y_true_levels1 = []
y_pred_levels1 = []

with torch.no_grad():
    for i, batch_data in enumerate(val_dataloader1):
        if batch_data is None:
            continue  # Skip empty batches
        inputs = batch_data["image"].to(device)
        condition_labels = batch_data["condition"].to(device)
        level_labels = batch_data["level"].to(device)

        condition_logits, level_logits = model1(inputs)
        condition_preds = torch.sigmoid(condition_logits) > 0.5
        level_preds = torch.sigmoid(level_logits) > 0.5

        y_true_conditions1.extend(condition_labels.cpu().numpy())
        y_pred_conditions1.extend(condition_preds.cpu().numpy())
        y_true_levels1.extend(level_labels.cpu().numpy())
        y_pred_levels1.extend(level_preds.cpu().numpy())

logger.info("First Dataset - Condition Classification Report:")
logger.info("\n" + classification_report(y_true_conditions1, y_pred_conditions1))

logger.info("First Dataset - Level Classification Report:")
logger.info("\n" + classification_report(y_true_levels1, y_pred_levels1))

# Initialize lists to store metrics for the second model
train_losses2 = []
val_losses2 = []
train_accuracies2 = []
val_accuracies2 = []

# Training loop for the second model
logger.info("Starting training for the second model...")
for epoch in range(num_epochs):
    start_time = time.time()
    model2.train()
    epoch_loss = 0
    correct_train_condition = 0
    correct_train_level = 0
    total_train = 0

    for i, batch_data in enumerate(train_dataloader2):
        if batch_data is None:
            continue  # Skip empty batches
        inputs = batch_data["image"].to(device)
        condition_labels = batch_data["condition"].to(device)
        level_labels = batch_data["level"].to(device)

        optimizer2.zero_grad()
        condition_logits, level_logits = model2(inputs)
        condition_loss = criterion(condition_logits, condition_labels)
        level_loss = criterion(level_logits, level_labels)
        loss = condition_loss + level_loss
        loss.backward()
        optimizer2.step()
        epoch_loss += loss.item()

        condition_preds = torch.sigmoid(condition_logits) > 0.4
        level_preds = torch.sigmoid(level_logits) > 0.4
        correct_train_condition += (condition_preds ==
                                    condition_labels).sum().item()
        correct_train_level += (level_preds == level_labels).sum().item()
        total_train += condition_labels.numel()

    train_loss = epoch_loss / len(train_dataloader2)
    train_accuracy = (correct_train_condition +
                      correct_train_level) / (2 * total_train)
    train_losses2.append(train_loss)
    train_accuracies2.append(train_accuracy)

    logger.info(
        f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss}, Accuracy: {train_accuracy}")

    # Validation
    model2.eval()
    val_loss = 0
    correct_val_condition = 0
    correct_val_level = 0
    total_val = 0
    with torch.no_grad():
        for i, batch_data in enumerate(val_dataloader2):
            if batch_data is None:
                continue  # Skip empty batches
            inputs = batch_data["image"].to(device)
            condition_labels = batch_data["condition"].to(device)
            level_labels = batch_data["level"].to(device)

            condition_logits, level_logits = model2(inputs)
            condition_loss = criterion(condition_logits, condition_labels)
            level_loss = criterion(level_logits, level_labels)
            loss = condition_loss + level_loss
            val_loss += loss.item()

            condition_preds = torch.sigmoid(condition_logits) > 0.4
            level_preds = torch.sigmoid(level_logits) > 0.4

            correct_val_condition += (condition_preds ==
                                      condition_labels).sum().item()
            correct_val_level += (level_preds == level_labels).sum().item()
            total_val += condition_labels.numel()

    val_loss /= len(val_dataloader2)
    val_accuracy = (correct_val_condition +
                    correct_val_level) / (2 * total_val)
    val_losses2.append(val_loss)
    val_accuracies2.append(val_accuracy)

    logger.info(f"Validation Loss: {val_loss}, Accuracy: {val_accuracy}")

    early_stopping2(val_loss, model2, checkpoint_path2)
    if early_stopping2.early_stop:
        logger.info("Early stopping")
        break

# Load the best model for the second dataset
model2.load_state_dict(torch.load(checkpoint_path2))

# Final evaluation for the second model
logger.info("Starting final evaluation for the second model...")
model2.eval()
y_true_conditions2 = []
y_pred_conditions2 = []
y_true_levels2 = []
y_pred_levels2 = []

with torch.no_grad():
    for i, batch_data in enumerate(val_dataloader2):
        if batch_data is None:
            continue  # Skip empty batches
        inputs = batch_data["image"].to(device)
        condition_labels = batch_data["condition"].to(device)
        level_labels = batch_data["level"].to(device)

        condition_logits, level_logits = model2(inputs)
        condition_preds = torch.sigmoid(condition_logits) > 0.4
        level_preds = torch.sigmoid(level_logits) > 0.4

        y_true_conditions2.extend(condition_labels.cpu().numpy())
        y_pred_conditions2.extend(condition_preds.cpu().numpy())
        y_true_levels2.extend(level_labels.cpu().numpy())
        y_pred_levels2.extend(level_preds.cpu().numpy())

logger.info("Second Dataset - Condition Classification Report:")
logger.info("\n" + classification_report(y_true_conditions2, y_pred_conditions2))

logger.info("Second Dataset - Level Classification Report:")
logger.info("\n" + classification_report(y_true_levels2, y_pred_levels2))

# Plot metrics for the first model
epochs = range(1, len(train_losses1) + 1)
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(epochs, train_losses1, 'b', label='Training Loss 1')
plt.plot(epochs, val_losses1, 'r', label='Validation Loss 1')
plt.title('Training and Validation Loss for First Model')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(epochs, train_accuracies1, 'b', label='Training Accuracy 1')
plt.plot(epochs, val_accuracies1, 'r', label='Validation Accuracy 1')
plt.title('Training and Validation Accuracy for First Model')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plot_path1 = os.path.join(output_dir, 'training_metrics_1.png')
plt.savefig(plot_path1)
logger.info(f"Saved training metrics plot for first model to {plot_path1}")
plt.close()

# Plot metrics for the second model
epochs = range(1, len(train_losses2) + 1)
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(epochs, train_losses2, 'b', label='Training Loss 2')
plt.plot(epochs, val_losses2, 'r', label='Validation Loss 2')
plt.title('Training and Validation Loss for Second Model')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(epochs, train_accuracies2, 'b', label='Training Accuracy 2')
plt.plot(epochs, val_accuracies2, 'r', label='Validation Accuracy 2')
plt.title('Training and Validation Accuracy for Second Model')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plot_path2 = os.path.join(output_dir, 'training_metrics_2.png')
plt.savefig(plot_path2)
logger.info(f"Saved training metrics plot for second model to {plot_path2}")
plt.close()

# Verify the saved artifacts
if os.path.exists(checkpoint_path1) and os.path.exists(plot_path1) and os.path.exists(checkpoint_path2) and os.path.exists(plot_path2):
    logger.info("All artifacts for both models saved successfully.")
else:
    logger.error("Some artifacts for the models are missing.")
#
#