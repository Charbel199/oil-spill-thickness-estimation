import numpy as np
from PIL import Image


def save_np_as_immage(arr, name) -> None:
    im = Image.fromarray(arr)
    im.save(f'{name}.jpeg')


def save_np(arr, name):
    np.save(f'{name}.npy', arr)  # save


def load_np(name) -> np.ndarray:
    return np.load(f'{name}.npy')  # load
