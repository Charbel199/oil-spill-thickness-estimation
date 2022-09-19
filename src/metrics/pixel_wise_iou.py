from metrics.metrics_helper import pixel_wise_metrics_evaluation, get_tp, get_fn


# iou = Intersection / Union
def pixel_wise_iou(y_true, y_pred, per_label = False):
    return pixel_wise_metrics_evaluation(y_true, y_pred, _iou, per_label)


def _iou(y_true, y_pred):
    intersection = (y_true & y_pred).sum()
    union = (y_true | y_pred).sum()
    return (intersection / union) if union != 0 else 0
