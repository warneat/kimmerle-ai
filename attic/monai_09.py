import os
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from monai.transforms import Compose, ScaleIntensity, EnsureType, Resize
from monai.networks.nets import DenseNet121
from pydicom import dcmread
import numpy as np
import matplotlib.pyplot as plt
import logging

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths using environment variables
csv_file = os.getenv("CSV_FILE", "./cleaned_train_label_coordinates.csv")
data_dir = os.getenv("DATA_DIR", "./small_set/train_images")
save_dir = os.getenv("SAVE_DIR", "./saved_models_and_plots")
os.makedirs(save_dir, exist_ok=True)

# Load CSV file
logger.info("Loading CSV file...")
data = pd.read_csv(csv_file)

# Filter necessary columns
data = data[["study_id", "series_id", "instance_number", "condition", "level"]]

# Split the data into training and validation sets
train_data, val_data = train_test_split(
    data, test_size=0.8, random_state=42, stratify=data[['condition', 'level']])

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

        # Manually add channel dimension
        image = np.expand_dims(image, axis=0)

        if self.transform:
            image = self.transform(image)

        condition = self.condition_to_index[row["condition"]]
        level = self.level_to_index[row["level"]]

        return {"image": image, "condition": condition, "level": level}


# Define transforms
transforms = Compose([
    ScaleIntensity(),
    Resize((320, 320)),  # Resize all images to 320x320
    EnsureType(data_type="tensor")
])

# Create datasets and dataloaders for training and validation
logger.info("Creating datasets and dataloaders...")
train_dataset = SpineDataset(train_data, data_dir, transform=transforms)
val_dataset = SpineDataset(val_data, data_dir, transform=transforms)

# Hyperparameters
batch_size = 256  # Larger batch size for efficient CPU usage
learning_rate = 5e-4  # Adjusted learning rate
num_epochs = 80  # Increased number of epochs

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
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=learning_rate)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Initialize lists to store loss values
train_losses = []
val_losses = []

# Training loop
logger.info("Starting training...")
for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0
    for i, batch_data in enumerate(train_dataloader):
        logger.info(f"Processing batch {i + 1}/{len(train_dataloader)}")
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

    train_losses.append(epoch_loss / len(train_dataloader))
    logger.info(
        f"Epoch {epoch + 1}/{num_epochs}, Train Loss: {epoch_loss / len(train_dataloader)}")

    # Validation loss
    model.eval()
    val_epoch_loss = 0
    with torch.no_grad():
        for i, batch_data in enumerate(val_dataloader):
            inputs = batch_data["image"].to(device)
            condition_labels = batch_data["condition"].to(device)
            level_labels = batch_data["level"].to(device)

            condition_logits, level_logits = model(inputs)
            condition_loss = criterion(condition_logits, condition_labels)
            level_loss = criterion(level_logits, level_labels)
            loss = condition_loss + level_loss
            val_epoch_loss += loss.item()

    val_losses.append(val_epoch_loss / len(val_dataloader))
    logger.info(
        f"Epoch {epoch + 1}/{num_epochs}, Val Loss: {val_epoch_loss / len(val_dataloader)}")

# Save the model
model_save_path = os.path.join(save_dir, "spine_stenosis_classifier.pth")
torch.save({
    'epoch': epoch + 1,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'loss': epoch_loss / len(train_dataloader),
    'batch_size': batch_size,
    'learning_rate': learning_rate,
    'num_epochs': num_epochs
}, model_save_path)
logger.info(f"Model saved to {model_save_path}")

# Save the loss values for plotting later
loss_log_path = os.path.join(save_dir, "loss_log.csv")
loss_data = pd.DataFrame({"epoch": range(1, num_epochs + 1), "train_loss": train_losses, "val_loss": val_losses,
                          "batch_size": batch_size, "learning_rate": learning_rate, "num_epochs": num_epochs})
loss_data.to_csv(loss_log_path, index=False)
logger.info(f"Loss log saved to {loss_log_path}")

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
logger.info("Condition Classification Report:")
logger.info("\n" + classification_report(y_true_conditions,
            y_pred_conditions, target_names=val_dataset.conditions))

logger.info("Level Classification Report:")
logger.info("\n" + classification_report(y_true_levels,
            y_pred_levels, target_names=val_dataset.levels))

# Plotting the loss values
loss_data = pd.read_csv(loss_log_path)

plt.figure()
plt.plot(loss_data["epoch"], loss_data["train_loss"], label="Train Loss")
plt.plot(loss_data["epoch"], loss_data["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()

# Adding hyperparameters information to the plot
hyperparams_info = f'Batch Size: {batch_size}, Learning Rate: {learning_rate}, Epochs: {num_epochs}'
plt.annotate(hyperparams_info, xy=(0.5, 0.01),
             xycoords='figure fraction', ha='center', fontsize=8)

plot_save_path = os.path.join(save_dir, "training_validation_loss.png")
plt.savefig(plot_save_path)
logger.info(f"Loss plot saved to {plot_save_path}")
