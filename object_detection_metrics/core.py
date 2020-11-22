import numpy as np


def iou_2d(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    Calculate intersection over union in 2D space

    :param box_a:  bounding box with format xyxy
    :param box_b:  bounding box with format xyxy
    :return: iou score in [0.0 - 1.0]
    """
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    intersection = max(0.0, (x_b - x_a)) * max(0.0, (y_b - y_a))

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    union = area_a + area_b - intersection

    return intersection / union


def precision_binary(ground_truth: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate precision for 2 classes:
    PRE = TP / (TP + FP)

    :param ground_truth: array of actual class labels
    :param predicted: array of predicted class labels
    :return: score in range [0.0 - 1.0]
    """
    tp = np.count_nonzero(np.multiply(ground_truth, predicted))
    if tp == 0:
        return 0
    fp = sum(true == 0 and pred == 1 for true, pred in zip(ground_truth, predicted))
    return tp / (tp + fp)


def precision_micro(ground_truth: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate precision for k classes using one-vs-all principe:
    PRE = TP_1 + TP_2 + ... + TP_k / (TP_1 + TP_2 + ... + TP_k + FP_1 + FP_2 + ... + FP_k)

    :param ground_truth: array of actual class labels
    :param predicted: array of predicted class labels
    :return: score in range [0.0 - 1.0]
    """
    class_ids = np.union1d(np.unique(ground_truth), np.unique(predicted))
    tp = 0
    fp = 0

    for class_id in class_ids:
        tp += sum(true == class_id and pred == class_id for true, pred in zip(ground_truth, predicted))
        fp += sum(true != class_id and pred == class_id for true, pred in zip(ground_truth, predicted))

    if tp == 0:
        return 0

    return tp / (tp + fp)


def precision_macro(ground_truth: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate precision for k classes using average:
    PRE = PRE_1 + PRE_2 + ... + PRE_k / k

    :param ground_truth: array of actual class labels
    :param predicted: array of predicted class labels
    :return: score in range [0.0 - 1.0]
    """
    class_ids = np.union1d(np.unique(ground_truth), np.unique(predicted))

    sum_pre = 0.0

    for class_id in class_ids:
        tp = sum(true == class_id and pred == class_id for true, pred in zip(ground_truth, predicted))
        fp = sum(true != class_id and pred == class_id for true, pred in zip(ground_truth, predicted))
        if tp != 0:
            sum_pre += tp / (tp + fp)

    return sum_pre / len(class_ids)


def recall_binary(ground_truth: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate recall for 2 classes:
    REC = TP / (TP + FN)

    :param ground_truth: array of actual class labels
    :param predicted: array of predicted class labels
    :return: score in range [0.0 - 1.0]
    """
    tp = np.count_nonzero(np.multiply(ground_truth, predicted))
    if tp == 0:
        return 0
    fn = sum(true == 1 and pred == 0 for true, pred in zip(ground_truth, predicted))
    return tp / (tp + fn)


def recall_micro(ground_truth: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate recall for k classes using one-vs-all principe:
    REC = TP_1 + TP_2 + ... + TP_k / (TP_1 + TP_2 + ... + TP_k + FN_1 + FN_2 + ... + FN_k)

    :param ground_truth: array of actual class labels
    :param predicted: array of predicted class labels
    :return: score in range [0.0 - 1.0]
    """
    class_ids = np.union1d(np.unique(ground_truth), np.unique(predicted))
    tp = 0
    fn = 0

    for class_id in class_ids:
        tp += sum(true == class_id and pred == class_id for true, pred in zip(ground_truth, predicted))
        fn += sum(true == class_id and pred != class_id for true, pred in zip(ground_truth, predicted))

    if tp == 0:
        return 0

    return tp / (tp + fn)


def recall_macro(ground_truth: np.ndarray, predicted: np.ndarray) -> float:
    """
    Calculate precision for k classes using average:
    REC = REC_1 + PRE_2 + ... + PRE_k / k)

    :param ground_truth: array of actual class labels
    :param predicted: array of predicted class labels
    :return: score in range [0.0 - 1.0]
    """
    class_ids = np.union1d(np.unique(ground_truth), np.unique(predicted))

    sum_rec = 0.0

    for class_id in class_ids:
        tp = sum(true == class_id and pred == class_id for true, pred in zip(ground_truth, predicted))
        fn = sum(true == class_id and pred != class_id for true, pred in zip(ground_truth, predicted))
        if tp != 0:
            sum_rec += tp / (tp + fn)

    return sum_rec / len(class_ids)
