from network import Network
import torch
from data_loader import StreetviewDataset
from utils import display_image, plot_location_predictions, plot_losses

BATCH_SIZE = 16
NUM_EPOCHS = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device {device}")

dataset = StreetviewDataset(
    "data/coords.csv",
    "data",
)

train_size, test_size = int(0.9 * len(dataset)), int(0.1 * len(dataset))
img, label = dataset[0]

network = Network(img.shape, len(label)).to(device)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(network.parameters(), lr=1e-5)

epoch_losses = []

for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch} - Training")
    for i in range(0, train_size, BATCH_SIZE):
        print(i)
        batch_images, batch_labels = [], []
        for j in range(BATCH_SIZE):
            img, label = dataset[i + j]
            batch_images.append(img)
            batch_labels.append(label)
        batch_images = torch.stack(batch_images).float().to(device)
        batch_labels = torch.stack(batch_labels).float().to(device)

        optimizer.zero_grad()
        result = network.forward(batch_images)
        loss = loss_fn(result, batch_labels)
        loss.backward()
        optimizer.step()
    print("Epoch {epoch} - Testing")
    with torch.no_grad():
        test_images, test_labels = [], []
        for i in range(train_size, len(dataset)):
            img, label = dataset[i]
            test_images.append(img)
            test_labels.append(label)
        test_images = torch.stack(test_images).float().to(device)
        test_labels = torch.stack(test_labels).float().to(device)
        result = network.forward(test_images)
        loss = loss_fn(result, test_labels)
        epoch_losses.append(loss.item())
        print(f"Epoch {epoch} - Loss: {loss.item()}")
        # plot_losses(epoch_losses)

plot_losses(epoch_losses)
plot_location_predictions(result.cpu(), test_labels.cpu())


display_image(img)


