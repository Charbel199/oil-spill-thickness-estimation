from metrics.metrics_helper import pixel_wise_metrics_evaluation, get_tp, get_fp


# Precision = TP / (TP + FP)
def pixel_wise_precision(y_true, y_pred):
    return pixel_wise_metrics_evaluation(y_true, y_pred, _precision)


def _precision(y_true, y_pred):
    tp = get_tp(y_true, y_pred)
    fp = get_fp(y_true, y_pred)
    precision = (tp / (tp + fp)) if (tp + fp) != 0 else 0
    return precision
