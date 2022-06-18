import numpy as np


def iou_coefficient(y_true, y_pred):

    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)

    iou_score = np.sum(intersection) / np.sum(union)

    return iou_score
