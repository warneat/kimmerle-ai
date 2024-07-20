import os
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from monai.transforms import Compose, LoadImage, ScaleIntensity, EnsureType, Resize
from monai.networks.nets import DenseNet121
from pydicom import dcmread
import numpy as np

# Define paths
csv_file = "/Users/dominikkimmerle/Documents/Studium/Master_01/ai-in-python-kimmerle/cleaned_train_label_coordinates.csv"
data_dir = "/Users/dominikkimmerle/Documents/Studium/Master_01/ai-in-python-kimmerle/small_set/train_images"

# Load CSV file
data = pd.read_csv(csv_file)

# Filter necessary columns
data = data[["study_id", "series_id", "instance_number", "condition", "level"]]

# Create a custom dataset


class SpineDataset(Dataset):
    def __init__(self, data, data_dir, transform=None):
        self.data = data
        self.data_dir = data_dir
        self.transform = transform
        self.conditions = sorted(data['condition'].unique())
        self.levels = sorted(data['level'].unique())
        self.condition_to_index = {cond: idx for idx,
                                   cond in enumerate(self.conditions)}
        self.level_to_index = {lvl: idx for idx, lvl in enumerate(self.levels)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = os.path.join(self.data_dir, str(row["study_id"]), str(
            row["series_id"]), f'{row["instance_number"]}.dcm')
        image = dcmread(image_path).pixel_array.astype(np.float32)

        if self.transform:
            image = self.transform(image)

        condition = self.condition_to_index[row["condition"]]
        level = self.level_to_index[row["level"]]

        # Expand dimensions to add channel dimension
        image = np.expand_dims(image, axis=0)

        return {"image": image, "condition": condition, "level": level}


# Define transforms
transforms = Compose([
    ScaleIntensity(),
    Resize((320, 320)),  # Resize all images to 320x320
    EnsureType()
])

# Create dataset and dataloader
dataset = SpineDataset(data, data_dir, transform=transforms)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Define the model


class SpineStenosisClassifier(nn.Module):
    def __init__(self, num_conditions, num_levels):
        super(SpineStenosisClassifier, self).__init__()
        self.model = DenseNet121(
            spatial_dims=2, in_channels=1, out_channels=num_conditions + num_levels)

    def forward(self, x):
        x = self.model(x)
        condition_logits, level_logits = torch.split(
            x, [num_conditions, num_levels], dim=1)
        return condition_logits, level_logits


num_conditions = len(dataset.conditions)
num_levels = len(dataset.levels)
model = SpineStenosisClassifier(num_conditions, num_levels)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=1e-4)
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for batch_data in dataloader:
        inputs = batch_data["image"].to(device)
        condition_labels = batch_data["condition"].to(device)
        level_labels = batch_data["level"].to(device)

        optimizer.zero_grad()
        condition_logits, level_logits = model(inputs)
        condition_loss = criterion(condition_logits, condition_labels)
        level_loss = criterion(level_logits, level_labels)
        loss = condition_loss + level_loss
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(dataloader)}")

# Validation (using the same dataset for simplicity)
model.eval()
y_true_conditions = []
y_pred_conditions = []
y_true_levels = []
y_pred_levels = []

with torch.no_grad():
    for batch_data in dataloader:
        inputs = batch_data["image"].to(device)
        condition_labels = batch_data["condition"].to(device)
        level_labels = batch_data["level"].to(device)

        condition_logits, level_logits = model(inputs)
        condition_preds = torch.argmax(condition_logits, dim=1)
        level_preds = torch.argmax(level_logits, dim=1)

        y_true_conditions.extend(condition_labels.cpu().numpy())
        y_pred_conditions.extend(condition_preds.cpu().numpy())
        y_true_levels.extend(level_labels.cpu().numpy())
        y_pred_levels.extend(level_preds.cpu().numpy())

# Evaluate results
print("Condition Classification Report:")
print(classification_report(y_true_conditions,
      y_pred_conditions, target_names=dataset.conditions))

print("Level Classification Report:")
print(classification_report(y_true_levels,
      y_pred_levels, target_names=dataset.levels))
