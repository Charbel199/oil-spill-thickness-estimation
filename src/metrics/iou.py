import numpy as np


def iou_coefficient(y_true, y_pred):
    intersection = y_true == y_pred
    union = np.logical_or(y_true, y_pred)

    iou_score = np.count_nonzero(intersection) / np.count_nonzero(union)

    return iou_score
