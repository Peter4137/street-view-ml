import os
import pandas as pd
from torch.utils.data import Dataset
from torchvision.io import read_image

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
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file, header=None)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, f"{idx}.png")
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label