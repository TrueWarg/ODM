import numpy as np
from object_detection_metrics.core import Average
from typing import Tuple


def mAP_2d(ground_truth: np.ndarray,
           predicted: np.ndarray,
           confidences: np.ndarray,
           thresholds: Tuple = tuple(0.5),
           average=Average.MACRO
           ) -> float:
    pass


def mAP_3d(ground_truth: np.ndarray,
           predicted: np.ndarray,
           confidences: np.ndarray,
           thresholds: Tuple = tuple(0.5),
           average=Average.MACRO
           ) -> float:
    pass
