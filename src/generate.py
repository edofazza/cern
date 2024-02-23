import os

import numpy as np
import uuid
import torch
from typing import Callable


def generate_random_rgb_image(n, transform: Callable, width: int = 32, height: int = 32):
    os.mkdir('generated')
    for _ in range(n):
        # Generate random pixel values for each channel (R, G, B)
        red_channel = np.random.randint(0, 256, size=(height, width), dtype=np.uint8)
        green_channel = np.random.randint(0, 256, size=(height, width), dtype=np.uint8)
        blue_channel = np.random.randint(0, 256, size=(height, width), dtype=np.uint8)

        # Stack the channels to form the RGB image
        rgb_image = np.stack([red_channel, green_channel, blue_channel], axis=-1)
        print(rgb_image.transpose(2, 0, 1).shape)
        rgb_image = rgb_image.transpose(2, 0, 1)
        transform_image = transform(rgb_image.transpose(2, 0, 1))
        np.save(f'generated/{str(uuid.uuid4())[:16]}.npy', transform_image.numpy())
