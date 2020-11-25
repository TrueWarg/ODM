from enum import Enum
from typing import Tuple, Dict, Collection


class BoxMatchingPolicy(Enum):
    GREEDY = 'greedy'
    SOFT = 'soft'


class MAP2DCalculator:
    def __init__(self,
                 class_ids: Tuple,
                 iou_thresholds: Tuple = tuple(0.5),
                 policy=BoxMatchingPolicy.GREEDY
                 ):
        self.class_ids = class_ids
        self.iou_thresholds = iou_thresholds
        self.policy = policy

    def calculate_mAP(self,
                      ground_truth: Collection,
                      predicted: Collection
                      ) -> float:
        """
        Calculate mAP score. Params ground_truth and predicted must be have same length.

        :param ground_truth: images per numpy arrays of truth boxes: [x, y, x, y, id, difficult, crowd]}
        :param predicted: images per numpy arrays of predicted boxes: [x, y, x, y, id, confidence]}
        :return: mAP score in [0.0 - 1.0]
        """
        images_count = len(ground_truth)
        for _ in range(images_count):
            for class_id in self.class_ids:
                pass
