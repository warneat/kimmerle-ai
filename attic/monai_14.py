import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tv_transforms
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from pydicom import dcmread
from PIL import Image
import logging
from monai.networks.nets import UNet


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

    def resize_image(self, image):
        img_pil = Image.fromarray(image)
        resized_image = self.transform(img_pil)
        return np.array(resized_image)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        series_dir = os.path.join(self.data_dir, str(
            row["study_id"]), str(row["series_id"]))
        image_files = sorted([os.path.join(series_dir, f)
                             for f in os.listdir(series_dir) if f.endswith('.dcm')])

        # Select or augment to have exactly 15 images
        if len(image_files) < 15:
            while len(image_files) < 15:
                image_files.extend(image_files[:15 - len(image_files)])
        elif len(image_files) > 15:
            start = (len(image_files) - 15) // 2
            image_files = image_files[start:start + 15]

        images = [self.resize_image(
            dcmread(f).pixel_array.astype(np.float32)) for f in image_files]
        images = [tv_transforms.ToTensor()(img).squeeze()
                  for img in images]  # Ensure no extra dimensions

        # Combine 15 images into a single tensor
        combined_image = torch.stack(images, dim=0)  # Shape: [15, 320, 320]
        combined_image = combined_image.unsqueeze(
            1)  # Shape: [15, 1, 320, 320]
        combined_image = combined_image.permute(
            1, 0, 2, 3)  # Shape: [1, 15, 320, 320]

        condition = self.condition_to_index[row["condition"]]
        level = self.level_to_index[row["level"]]

        return {"image": combined_image, "condition": condition, "level": level}


def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)

    # Define paths
    csv_file = "./cleaned_train_label_coordinates.csv"
    data_dir = "./small_set/train_images"

    # Load CSV file
    logger.info("Loading CSV file...")
    data = pd.read_csv(csv_file)

    # Filter necessary columns
    data = data[["study_id", "series_id",
                 "instance_number", "condition", "level"]]

    # Check for available images in each series and select up to 3 series per study with at least 15 images
    logger.info("Checking for available images in each series...")

    def filter_series(data, base_dir):
        valid_series = []
        grouped = data.groupby(['study_id', 'series_id'])
        for (study_id, series_id), group in grouped:
            series_dir = os.path.join(base_dir, str(study_id), str(series_id))
            if os.path.isdir(series_dir):
                images = [os.path.join(series_dir, f) for f in os.listdir(
                    series_dir) if f.endswith('.dcm')]
                if len(images) >= 15:
                    selected_images = select_middle_images(images, 15)
                else:
                    selected_images = images.copy()
                    while len(selected_images) < 15:
                        augmented_images = augment_images(images)
                        selected_images.extend(augmented_images)
                    selected_images = selected_images[:15]
                valid_series.append(
                    (study_id, series_id, selected_images, group))
        return valid_series

    def select_middle_images(images, target_count):
        middle = len(images) // 2
        start = max(0, middle - target_count // 2)
        end = start + target_count
        return images[start:end]

    def augment_images(images):
        augmented_images = []
        for image_path in images:
            image = dcmread(image_path).pixel_array.astype(np.float32)
            zoomed_image = zoom_image(image)
            augmented_images.append(zoomed_image)
            if len(images) + len(augmented_images) >= 15:
                break
        return augmented_images[:15 - len(images)]

    def zoom_image(image, zoom_factor=1.1):
        height, width = image.shape
        new_height, new_width = int(
            height * zoom_factor), int(width * zoom_factor)
        zoomed_image = np.zeros_like(image)

        # Calculate cropping box
        crop_height = (new_height - height) // 2
        crop_width = (new_width - width) // 2

        # Crop the center of the image
        zoomed_image = image[crop_height:crop_height +
                             height, crop_width:crop_width + width]

        return zoomed_image

    filtered_series = filter_series(data, data_dir)
    if not filtered_series:
        logger.error("No series found with at least 15 images. Exiting.")
        exit(1)

    # Limit to 3 series per study
    study_series_count = {}
    selected_series = []
    for study_id, series_id, images, group in filtered_series:
        if study_id not in study_series_count:
            study_series_count[study_id] = 0
        if study_series_count[study_id] < 3:
            group = group.head(len(images))
            selected_series.append(group)
            study_series_count[study_id] += 1

    valid_series_df = pd.concat(selected_series)
    logger.info(
        f"Found {valid_series_df['series_id'].nunique()} valid series with at least 15 images.")

    valid_series_df = valid_series_df.groupby(['study_id', 'series_id']).apply(
        lambda x: x.head(15)).reset_index(drop=True)

    # Split the data into training and validation sets
    logger.info("Splitting data into training and validation sets...")
    train_data, val_data = train_test_split(
        valid_series_df, test_size=0.2, random_state=42, stratify=valid_series_df[['condition', 'level']])

    # Define transforms
    transform = tv_transforms.Compose([
        tv_transforms.Resize((320, 320)),  # Adjusted size
        # Ensure single channel
        tv_transforms.Grayscale(num_output_channels=1),
        tv_transforms.ToTensor(),
    ])

    # Create datasets and dataloaders for training and validation
    logger.info("Creating datasets and dataloaders...")
    train_dataset = SpineDataset(train_data, data_dir, transform=transform)
    val_dataset = SpineDataset(val_data, data_dir, transform=transform)

    batch_size = 16  # Adjust batch size as needed

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)

    # Define the model
    class SpineStenosisClassifier(nn.Module):
        def __init__(self, num_conditions, num_levels):
            super(SpineStenosisClassifier, self).__init__()
            self.model = UNet(
                spatial_dims=3,
                in_channels=1,
                out_channels=num_conditions + num_levels,
                channels=(16, 32, 64, 128, 256),
                strides=(2, 2, 2, 2),
                num_res_units=2,
            )

        def forward(self, x):
            # Expecting x to be of shape [batch_size, 1, depth, height, width]
            x = self.model(x)
            condition_logits, level_logits = torch.split(
                x, [num_conditions, num_levels], dim=1)
            # Flatten the logits for classification
            condition_logits = condition_logits.view(x.size(0), -1)
            level_logits = level_logits.view(x.size(0), -1)
            return condition_logits, level_logits

    num_conditions = len(train_dataset.conditions)
    num_levels = len(train_dataset.levels)
    model = SpineStenosisClassifier(num_conditions, num_levels)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=5e-5)  # Adjusted learning rate
    num_epochs = 25  # Number of epochs
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
            condition_labels = batch_data["condition"].to(device)
            level_labels = batch_data["level"].to(device)

            optimizer.zero_grad()
            condition_logits, level_logits = model(inputs)

            #logger.info(f"Condition logits shape: {condition_logits.shape}")
            #logger.info(f"Level logits shape: {level_logits.shape}")
            #logger.info(f"Condition labels shape: {condition_labels.shape}")
            #logger.info(f"Level labels shape: {level_labels.shape}")

            # Ensure the logits are the correct shape for classification loss
            condition_loss = criterion(condition_logits, condition_labels)
            level_loss = criterion(level_logits, level_labels)

            loss = condition_loss + level_loss
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            logger.info(
                f"Batch {i + 1}/{len(train_dataloader)} processed. Loss: {loss.item()}")

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


if __name__ == "__main__":
    main()
