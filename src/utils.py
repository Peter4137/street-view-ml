import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
from geodatasets import get_path

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


def plot_location_predictions(predictions: np.ndarray, locations: np.ndarray) -> None:
    """Plot a location on a world map
    
    Args:
        location (np.ndarray): Location to plot. Expected shape is (2,)
    """
    world = gpd.read_file(get_path("naturalearth.land"))
    world.plot()

    for pred, loc in zip(predictions, locations):
        normalised_location = location_from_normalised(loc)
        normalised_prediction = location_from_normalised(pred)
        plt.plot([normalised_location[1], normalised_prediction[1]], [normalised_location[0], normalised_prediction[0]], marker="o", linestyle="--")

    plt.show()
    # plt.draw()
    # plt.pause(0.001)
