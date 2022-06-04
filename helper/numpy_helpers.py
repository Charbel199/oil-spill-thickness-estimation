import numpy as np


def save_np(arr, name):
    np.save(f'{name}.npy', arr)  # save


def load_np(name) -> np.ndarray:
    return np.load(f'{name}.npy')  # load
