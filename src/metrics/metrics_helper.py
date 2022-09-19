import numpy as np
from helper.general_helpers import filter_classes, avg_list


def pixel_wise_metrics_evaluation(y_true, y_pred, func, per_label = False):
    classes = np.unique(y_true, return_counts=False)
    metrics = []
    for c in classes:
        metrics.append(func(filter_classes(y_true, c), filter_classes(y_pred, c)))
    if not per_label:
        return avg_list(metrics)
    return metrics

def get_tp(y_true, y_pred):
    return ((y_true == 1) & (y_pred == 1)).sum()


def get_fp(y_true, y_pred):
    return ((y_true == 0) & (y_pred == 1)).sum()


def get_tn(y_true, y_pred):
    return ((y_true == 0) & (y_pred == 0)).sum()


def get_fn(y_true, y_pred):
    return ((y_true == 1) & (y_pred == 0)).sum()
