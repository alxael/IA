import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

# loader for the training and validation data
class TrainingDataset(Dataset):
    def __init__(self, csv_file, images_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # load image and label
        row = self.data.iloc[idx]
        image_id = row['image_id']
        label = int(row['label'])
        img_path = os.path.join(self.images_dir, f"{image_id}.png")
        image = Image.open(img_path).convert('RGB')

        # apply transformation to image
        if self.transform:
            image = self.transform(image)

        return image, label

# loader for the test data
class TestDataset(Dataset):
    def __init__(self, csv_file, images_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.images_dir = images_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # load image
        row = self.data.iloc[idx]
        image_id = row['image_id']
        img_path = os.path.join(self.images_dir, f"{image_id}.png")
        image = Image.open(img_path).convert('RGB')

        # apply transformation to image        
        if self.transform:
            image = self.transform(image)

        return image, image_id
