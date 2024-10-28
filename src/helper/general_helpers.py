import numpy as np
from typing import List



def filter_classes(arr: np.ndarray, cls: int) -> np.ndarray:
    # Filter classes and only keep the specified class
    return (arr == cls).astype(int)


def avg_list(arr: List):
    return sum(arr) / len(arr)
