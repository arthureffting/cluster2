import cv2
import numpy as np
import torch

from scripts.utils.image_handling import tensmeyer_brightness


def apply_tensmeyer_brightness(images, sigma=30, **kwargs):
    random_state = np.random.RandomState(kwargs.get("random_seed", None))
    foreground = random_state.normal(0, sigma)
    background = random_state.normal(0, sigma)

    for img in images:
        if img.shape[2] != 3:
            print("wrong image depth")

    return [tensmeyer_brightness(img, foreground, background) for img in images]


def apply_random_color_rotation(images, **kwargs):
    random_state = np.random.RandomState(kwargs.get("random_seed", None))
    shift = random_state.randint(0, 255)
    new_images = []
    for img in images:
        try:
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            hsv[..., 0] = hsv[..., 0] + shift
            new_images.append(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))
        except Exception as e:
            new_images.append(img)
    return new_images


def rescale(images, size):
    return [cv2.resize(img, (size, size), interpolation=cv2.INTER_CUBIC) for img in images]


def transpose_to_torch(img, dtype=np.float32):
    img = img.transpose([2, 0, 1])[None, ...]
    img = img.astype(dtype)
    img = torch.from_numpy(img)
    img = img / 128.0 - 1.0
    return img
