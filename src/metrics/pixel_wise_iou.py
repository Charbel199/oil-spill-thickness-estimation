from metrics.metrics_helper import pixel_wise_metrics_evaluation, get_tp, get_fn


# iou = Intersection / Union
def pixel_wise_iou(y_true, y_pred):
    return pixel_wise_metrics_evaluation(y_true, y_pred, _iou)


def _iou(y_true, y_pred):
    return (y_true & y_pred).sum() / (y_true | y_pred).sum()
