import math
from network import Network
import torch
from dataset import StreetviewDataset
from torch.utils.data import DataLoader
from utils import display_image, plot_location_predictions, plot_losses

BATCH_SIZE = 64
TEST_BATCH_SIZE = 10
NUM_EPOCHS = 1
TEST_TRAIN_SPLIT = 0.9

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

dataset = StreetviewDataset(
    "data/coords.csv",
    "data",
)

training_dataset, test_dataset = torch.utils.data.random_split(
    dataset, 
    [math.ceil(TEST_TRAIN_SPLIT * len(dataset)), math.ceil((1-TEST_TRAIN_SPLIT) * len(dataset))]
)

training_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
testing_dataloader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=True)

img, label = dataset[0]
network = Network(img.shape, len(label)).to(device)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(network.parameters(), lr=1e-5)

epoch_losses = []

for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch} - Training")
    for i, (images, labels) in enumerate(training_dataloader):
        print(i)
        optimizer.zero_grad()
        result = network.forward(images.to(device))
        loss = loss_fn(result, labels.to(device))
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch} - Testing")
    with torch.no_grad():
        all_losses = []
        for images, labels in testing_dataloader:
            result = network.forward(images.to(device))
            loss = loss_fn(result, labels.to(device))
            all_losses.append(loss.item())
        avg_loss = sum(all_losses) / len(all_losses)
        epoch_losses.append(avg_loss)
        print(f"Epoch {epoch} - Loss: {avg_loss}")

print(labels, result)
plot_losses(epoch_losses)
plot_location_predictions(result.cpu(), labels.cpu(), images.cpu())

# TODO
# Improve hyperparameter tuning
# Use full dataset - try using larger images
# Change network architecture to improve results
# Add zones to partition world, new classifier for each zone
# Add readme


