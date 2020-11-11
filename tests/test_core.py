from object_detection_metrics.core import iou_2d
import unittest


class IoU2DTest(unittest.TestCase):
    def test_box_a_top_left_relative_box_b_case_1(self):
        # Arrange
        box_a = [0, 0, 4, 4]
        box_b = [2, 2, 6, 6]

        # Act
        result = iou_2d(box_a, box_b)

        print(result)
        print(2 * 2/(4 * 4 + 4 * 4 - 2 * 2))

        # Assert
        self.assertAlmostEquals(result, 2 * 2/(4 * 4 + 4 * 4 - 2 * 2), places=8)

    def test_box_a_top_left_relative_box_b_case_2(self):
        # Arrange
        box_a = [20, 20, 80, 80]
        box_b = [40, 40, 90, 90]

        # Act
        result = iou_2d(box_a, box_b)

        # Assert
        self.assertAlmostEquals(result, 40 * 40/(60 * 60 + 50 * 50 - 40 * 40), places=8)

    def test_box_a_top_left_relative_box_b_case_3(self):
        # Arrange
        box_a = [3, 7, 8, 9]
        box_b = [3, 7, 8, 10]

        # Act
        result = iou_2d(box_a, box_b)

        # Assert
        self.assertAlmostEquals(result, 5 * 2/(5 * 2 + 5 * 3 - 5 * 2), places=8)