from functools import cache
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
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir

    def mirror_image(self, image):
        """Mirror the image"""
        return torch.flip(image, [2])

    def transform_label(self, label):
        return torch.tensor([
            label[0] / 90,
            label[1] / 180,
        ]).float()
    
    def transform_image(self, image):
        return Resize((64, 64))(image).float()

    def __len__(self):
        return 2 * len(self.img_labels)

    def __getitem__(self, idx):
        idx, augmented = divmod(idx, 2)
        img_path = os.path.join(self.img_dir, f"{idx}.png")
        image = read_image(img_path)
        if augmented:
            image = self.mirror_image(image)
        label = torch.tensor(self.img_labels.iloc[idx])
        return self.transform_image(image), self.transform_label(label)