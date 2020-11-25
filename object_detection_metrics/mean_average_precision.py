from enum import Enum
from typing import Tuple, Dict


class BoxMatchingPolicy(Enum):
    GREEDY = 'greedy'
    SOFT = 'soft'


class MAP2DCalculator:
    def __init__(self,
                 class_count: int,
                 iou_thresholds: Tuple = tuple(0.5),
                 policy=BoxMatchingPolicy.GREEDY
                 ):
        self.class_count = class_count
        self.iou_thresholds = iou_thresholds
        self.policy = policy

    def calculate_mAP(self,
                      ground_truth: Dict,
                      predicted: Dict
                      ) -> float:
        """
        Calculate mAP score

        :param ground_truth: {image_id : numpy array of truth boxes: [x, y, x, y, id, difficult, crowd]}
        :param predicted: {image_id : numpy array of truth boxes: [x, y, x, y, id, confidence]}
        :return: mAP score in [0.0 - 1.0]
        """
        pass
