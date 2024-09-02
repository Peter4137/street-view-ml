from network import Network
import torch
from data_loader import StreetviewDataset
from utils import display_image

BATCH_SIZE = 1

dataset = StreetviewDataset(
    "data/coords.csv",
    "data",
    transform=None,
    target_transform=None
)
img, label = dataset[0]

network = Network(img.shape, len(label))

result = network.forward(torch.tensor(img).float())
print(result)


display_image(img)


