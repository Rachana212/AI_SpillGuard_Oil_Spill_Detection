import cv2
import numpy as np

def predict_oil_spill(image_np):
    """
    Dummy oil spill prediction function
    Replace this with real model later
    """

    # Resize image to fixed size
    img = cv2.resize(image_np, (256, 256))

    # Create dummy probability map
    prob_map = np.random.rand(256, 256)

    # Create binary mask
    mask = (prob_map > 0.6).astype(np.uint8)

    return prob_map, mask
