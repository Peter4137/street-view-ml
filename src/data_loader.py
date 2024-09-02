import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.transforms import Resize
import torch

class StreetviewDataset(Dataset):
    """Dataset for the streetview images
    Args:
        annotations_file (string): Path to the csv file with annotations.
        img_dir (string): Directory with all the images.
        transform (callable, optional): Optional transform to be applied
            on a sample.
        target_transform (callable, optional): Optional transform to be applied
            on a target.
    
    """
    def __init__(self, annotations_file, img_dir):
        self.img_labels = pd.read_csv(annotations_file, header=None)[:1000]
        self.img_dir = img_dir

    def transform_label(self, label):
        return torch.tensor([
            label[0] / 90,
            label[1] / 180,
        ])
    
    def transform_image(self, image):
        """Scale image to 64 by 64 pixels"""
        return Resize((64, 64))(image)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f"{idx}.png")
        image = read_image(img_path)
        label = torch.tensor(self.img_labels.iloc[idx])
        return self.transform_image(image), self.transform_label(label)