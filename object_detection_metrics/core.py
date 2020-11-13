from dataclasses import dataclass
from typing import List
from collections import Counter

import numpy as np


@dataclass()
class Box2D:
    left: float
    top: float
    right: float
    bottom: float


def box2d_from_xyxy(box_values) -> Box2D:
    return Box2D(
        left=box_values[0],
        top=box_values[1],
        right=box_values[2],
        bottom=box_values[3],
    )


def box2d_list_from_xyxy(box_values_list) -> List[Box2D]:
    return [box2d_from_xyxy(box) for box in box_values_list]


def box2d_from_xywh(box_values) -> Box2D:
    return Box2D(
        left=box_values[0],
        top=box_values[1],
        right=box_values[0] + box_values[2],
        bottom=box_values[1] + box_values[3],
    )


def box2d_list_from_xywh(box_values_list) -> List[Box2D]:
    return [box2d_from_xywh(box) for box in box_values_list]


def iou_2d(box_a: Box2D, box_b: Box2D) -> float:
    x_a = max(box_a.left, box_b.left)
    y_a = max(box_a.top, box_b.top)
    x_b = min(box_a.right, box_b.right)
    y_b = min(box_a.bottom, box_b.bottom)

    intersection = max(0.0, (x_b - x_a)) * max(0.0, (y_b - y_a))

    area_a = (box_a.right - box_a.left) * (box_a.bottom - box_a.top)
    area_b = (box_b.right - box_b.left) * (box_b.bottom - box_b.top)

    union = area_a + area_b - intersection

    return intersection / union


def precision_micro(ground_truth: np.ndarray, predicted: np.ndarray) -> float:
    tp = np.count_nonzero(np.multiply(ground_truth, predicted))
    if tp == 0:
        return 0
    fp = sum(true == 0 and pred == 1 for true, pred in zip(ground_truth, predicted))
    return tp / (tp + fp)


def precision_macro(ground_truth, predicted) -> float:
    return 0.0


def recall():
    pass
