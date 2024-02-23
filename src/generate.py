import os

import numpy as np
import uuid
import torch
from typing import Callable
from PIL import Image

def generate_random_rgb_image(n, transform: Callable, width: int = 32, height: int = 32):
    os.mkdir('generated')
    for _ in range(n):
        # Generate random pixel values for each channel (R, G, B)
        random_image = np.random.randint(0, 256, size=(height, width, 3), dtype=np.uint8)

        # Stack the channels to form the RGB image
        transformed_img = transform(random_image)
        numpy_array = np.array(transformed_img)
        np.save(f'generated/{str(uuid.uuid4())[:16]}.npy', numpy_array)
