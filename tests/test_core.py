from object_detection_metrics.core import iou_2d, Box2D
import unittest


class IoU2DTest(unittest.TestCase):
    def test_box_a_top_left_relative_box_b_case_1(self):
        # Arrange
        box_a = Box2D(0, 0, 4, 4)
        box_b = Box2D(2, 2, 6, 6)

        # Act
        result = iou_2d(box_a, box_b)

        # Assert
        self.assertAlmostEqual(result, 2 * 2 / (4 * 4 + 4 * 4 - 2 * 2), places=8)

    def test_box_a_top_left_relative_box_b_case_2(self):
        # Arrange
        box_a = Box2D(20, 20, 80, 80)
        box_b = Box2D(40, 40, 90, 90)

        # Act
        result = iou_2d(box_a, box_b)

        # Assert
        self.assertAlmostEqual(result, 40 * 40 / (60 * 60 + 50 * 50 - 40 * 40), places=8)

    def test_box_a_top_left_relative_box_b_case_3(self):
        # Arrange
        box_a = Box2D(3, 7, 8, 9)
        box_b = Box2D(3, 7, 8, 10)

        # Act
        result = iou_2d(box_a, box_b)

        # Assert
        self.assertAlmostEqual(result, 5 * 2 / (5 * 2 + 5 * 3 - 5 * 2), places=8)
