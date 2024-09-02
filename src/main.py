from network import Network
import torch
from data_loader import StreetviewDataset
from utils import display_image


dataset = StreetviewDataset(
    "data/coords.csv",
    "data",
    transform=None,
    target_transform=None
)

img, label = dataset[0]
display_image(img)
