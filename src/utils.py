import numpy as np
import matplotlib.pyplot as plt

def display_image(image: np.ndarray) -> None:
    """Display an image
    
    Args:
        image (np.ndarray): Image to display. Expected shape is (C, H, W)
    """
    valid_image = np.transpose(image, (1, 2, 0))
    plt.imshow(valid_image)
    plt.show()
