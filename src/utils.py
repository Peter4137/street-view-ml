import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from geodatasets import get_path
import matplotlib.patches as patches
import matplotlib.lines as mlines

# plt.ion()

def display_image(image: np.ndarray) -> None:
    """Display an image
    
    Args:
        image (np.ndarray): Image to display. Expected shape is (C, H, W)
    """
    valid_image = np.transpose(image, (1, 2, 0))
    plt.imshow(valid_image)
    plt.show()

def plot_losses(losses: list) -> None:
    """Plot the losses
    
    Args:
        losses (list): List of losses
    """
    plt.plot(losses)
    plt.show()

def location_from_normalised(location: np.ndarray) -> np.ndarray:
    """Convert a normalized location to a real location
    
    Args:
        location (np.ndarray): Normalized location. Expected shape is (2,)
    
    Returns:
        np.ndarray: Real location. Shape is (2,)
    """
    return np.array([
        location[0] * 90,
        location[1] * 180,
    ])


def plot_location_predictions(predictions: np.ndarray, locations: np.ndarray, images: np.ndarray) -> None:
    """Plot a location on a world map
    
    Args:
        location (np.ndarray): Location to plot. Expected shape is (2,)
    """
    _, axs = plt.subplot_mosaic(
        [["map" for _ in predictions],
        [f"image{i}" for i, _ in enumerate(predictions)]],
        layout="constrained",
    )

    world = gpd.read_file(get_path("naturalearth.land"))
    world.plot(ax=axs["map"], alpha=0.5)

    actual_marker = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
        markersize=10, label='Actual')
    predicted_marker = mlines.Line2D([], [], color='black', marker='x', linestyle='None',
        markersize=10, label='Predicted')
    
    axs["map"].legend(handles=[predicted_marker, actual_marker])

    for i, (pred, loc, image) in enumerate(zip(predictions, locations, images)):
        normalised_location = location_from_normalised(loc)
        normalised_prediction = location_from_normalised(pred)
        plot = axs["map"].plot([normalised_location[1], normalised_prediction[1]], [normalised_location[0], normalised_prediction[0]], linestyle="--")
        axs["map"].scatter(normalised_location[1], normalised_location[0], color=plot[0].get_color(), marker="o")
        axs["map"].scatter(normalised_prediction[1], normalised_prediction[0], color=plot[0].get_color(), marker="x")

        rect = patches.Rectangle(
            (0, 0), 
            image.shape[1], 
            image.shape[2] // 10, 
            linewidth=1, 
            facecolor=plot[0].get_color()
        )

        axs[f"image{i}"].imshow(np.transpose(image, (1, 2, 0)).int())
        axs[f"image{i}"].axis("off")
        axs[f"image{i}"].add_patch(rect)

    plt.show()