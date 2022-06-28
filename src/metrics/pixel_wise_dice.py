from metrics.metrics_helper import pixel_wise_metrics_evaluation, get_tp, get_fn, get_fp


# Dice = 2*TP / (2*TP + FN + FP)
def pixel_wise_dice(y_true, y_pred):
    return pixel_wise_metrics_evaluation(y_true, y_pred, _dice)


def _dice(y_true, y_pred):
    tp = get_tp(y_true, y_pred)
    fn = get_fn(y_true, y_pred)
    fp = get_fp(y_true, y_pred)
    dice = ((2 * tp) / (2 * tp + fn + fp)) if (2 * tp + fn + fp) != 0 else 0
    return dice
