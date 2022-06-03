import numpy as np


def save_array(arr, name):
    np.save(f'{name}.npy', arr)  # save


def load_np(name):
    return np.load(f'{name}.npy')  # load
