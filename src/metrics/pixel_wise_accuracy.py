from metrics.metrics_helper import pixel_wise_metrics_evaluation, get_tp, get_fp, get_fn, get_tn


# Accuracy = TP + TN / (TP + FP + TN + FN)
def pixel_wise_accuracy(y_true, y_pred):
    return pixel_wise_metrics_evaluation(y_true, y_pred, _accuracy)


def _accuracy(y_true, y_pred):
    tp = get_tp(y_true, y_pred)
    fp = get_fp(y_true, y_pred)
    tn = get_tn(y_true, y_pred)
    fn = get_fn(y_true, y_pred)
    accuracy = ((tp + tn) / (tp + fp + tn + fn)) if (tp + fp + tn + fn) != 0 else 0
    return accuracy
