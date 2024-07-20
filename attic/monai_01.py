import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from monai.transforms import LoadImage, EnsureChannelFirst, ScaleIntensity, EnsureType, Compose, Resize
from monai.data import Dataset, DataLoader
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure environment
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load annotations
logger.info("Loading annotations.")
annotations = pd.read_csv("path/to/train_series_descriptions.csv")
label_annotations = pd.read_csv("path/to/train_label_coordinates.csv")
logger.info("Merging annotations.")
annotations = pd.merge(annotations, label_annotations, on=['study_id', 'series_id'])

# Split data
logger.info("Splitting data into training, validation, and test sets.")
train_val_annotations, test_annotations = train_test_split(annotations, test_size=0.1, random_state=42)
train_annotations, val_annotations = train_test_split(train_val_annotations, test_size=0.22, random_state=42)

# Function to create dataset items
def create_dataset_items(root_dir, annotations):
    items = []
    for _, row in annotations.iterrows():
        img_path = os.path.join(root_dir, str(row['study_id']), str(row['series_id']), f"{row['instance_number']}.dcm")
        if not os.path.exists(img_path):
            logger.warning(f"File not found: {img_path}")
            continue
        logger.info(f"Adding image: {img_path}")
        label = row[['x', 'y']].values.astype('float32')
        items.append({"image": img_path, "label": label})
    return items

# Create dataset items
root_dir = "path/to/dicom/folders"
logger.info("Creating dataset items.")
train_items = create_dataset_items(root_dir, train_annotations)
val_items = create_dataset_items(root_dir, val_annotations)
test_items = create_dataset_items(root_dir, test_annotations)

# Log the number of items
logger.info(f"Number of training items: {len(train_items)}")
logger.info(f"Number of validation items: {len(val_items)}")
logger.info(f"Number of test items: {len(test_items)}")

# Define transforms
logger.info("Defining transforms.")
transforms = Compose([
    LoadImage(image_only=True, reader="PydicomReader"),
    EnsureChannelFirst(),
    ScaleIntensity(),
    Resize((224, 224)),
    EnsureType()
])

# Create datasets and dataloaders
logger.info("Creating MONAI datasets.")
train_dataset = Dataset(data=train_items, transform=transforms)
val_dataset = Dataset(data=val_items, transform=transforms)
test_dataset = Dataset(data=test_items, transform=transforms)
logger.info("Creating DataLoaders.")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the model
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(32 * 112 * 112, 512)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 32 * 112 * 112)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Instantiate the model and move it to the appropriate device
logger.info("Instantiating the model and moving it to the appropriate device.")
model = SimpleCNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training and validation loop
num_epochs = 10
best_val_loss = float('inf')

for epoch in range(num_epochs):
    logger.info(f"Epoch {epoch+1}/{num_epochs}")
    model.train()
    running_loss = 0.0

    for batch in train_loader:
        logger.info(f"Processing batch with input paths: {batch['image']}")
        inputs, labels = batch["image"].to(device), batch["label"].to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    epoch_loss = running_loss / len(train_dataset)
    logger.info(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {epoch_loss:.4f}')

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            logger.info(f"Processing validation batch with input paths: {batch['image']}")
            inputs, labels = batch["image"].to(device), batch["label"].to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)

    val_loss /= len(val_dataset)
    logger.info(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pth')
        logger.info("Model saved!")

logger.info('Training complete')

# Load the best model
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Generate predictions
logger.info("Generating predictions.")
predictions = []

with torch.no_grad():
    for batch in test_loader:
        logger.info(f"Processing test batch with input paths: {batch['image']}")
        inputs = batch["image"].to(device)
        outputs = model(inputs)
        predictions.extend(outputs.cpu().numpy())

# Format output
output_data = []

for idx, row in test_annotations.iterrows():
    study_id = row['study_id']
    series_id = row['series_id']
    instance_number = row['instance_number']
    condition = row['condition']
    level = row['level']
    row_id = f"{study_id}_{series_id}_{condition}_{level}".lower().replace(" ", "_")
    x, y = predictions[idx]
    output_data.append([row_id, x, y])

logger.info("Saving predictions to CSV.")
output_df = pd.DataFrame(output_data, columns=['row_id', 'x', 'y'])
output_df.to_csv('output_predictions.csv', index=False)
logger.info("Script complete.")
