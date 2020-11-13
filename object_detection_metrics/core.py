import numpy as np


def iou_2d(box_a: np.ndarray, box_b: np.ndarray) -> float:
    x_a = max(box_a[0], box_b[0])
    y_a = max(box_a[1], box_b[1])
    x_b = min(box_a[2], box_b[2])
    y_b = min(box_a[3], box_b[3])

    intersection = max(0.0, (x_b - x_a)) * max(0.0, (y_b - y_a))

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    union = area_a + area_b - intersection

    return intersection / union


def precision_micro(ground_truth: np.ndarray, predicted: np.ndarray) -> float:
    tp = np.count_nonzero(np.multiply(ground_truth, predicted))
    if tp == 0:
        return 0
    fp = sum(true == 0 and pred == 1 for true, pred in zip(ground_truth, predicted))
    return tp / (tp + fp)


def precision_macro() -> float:
    return 0.0


def recall_micro(ground_truth: np.ndarray, predicted: np.ndarray) -> float:
    tp = np.count_nonzero(np.multiply(ground_truth, predicted))
    if tp == 0:
        return 0
    fn = sum(true == 1 and pred == 0 for true, pred in zip(ground_truth, predicted))
    return tp / (tp + fn)
