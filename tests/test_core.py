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

    def test_box_a_top_right_relative_box_b_case_1(self):
        # Arrange
        box_a = Box2D(400, 30.5, 420, 50)
        box_b = Box2D(350, 35.7, 410, 52.01)

        # Act
        result = iou_2d(box_a, box_b)

        # Assert
        self.assertAlmostEqual(result, 10 * 14.3 / (20 * 19.5 + 60 * 16.31 - 10 * 14.3), places=8)

    def test_box_a_top_right_relative_box_b_case_2(self):
        # Arrange
        box_a = Box2D(20, 12, 36, 13.5)
        box_b = Box2D(20, 12.5, 36, 14)

        # Act
        result = iou_2d(box_a, box_b)

        # Assert
        self.assertAlmostEqual(result, 16 * 1 / (16 * 1.5 + 16 * 1.5 - 16 * 1), places=8)

    def test_box_a_top_right_relative_box_b_case_3(self):
        # Arrange
        box_a = Box2D(10, 10, 20, 20)
        box_b = Box2D(5, 20, 20, 30)

        # Act
        result = iou_2d(box_a, box_b)

        # Assert
        self.assertAlmostEqual(result, 0, places=8)

    def test_box_a_bottom_right_relative_box_b_case_1(self):
        # Arrange
        box_a = Box2D(4, 4, 10, 10)
        box_b = Box2D(0, 0, 5, 5)

        # Act
        result = iou_2d(box_a, box_b)

        # Assert
        self.assertAlmostEqual(result, 1 * 1 / (6 * 6 + 5 * 5 - 1 * 1), places=8)

    def test_box_a_bottom_right_relative_box_b_case_2(self):
        # Arrange
        box_a = Box2D(2.3, 3.7, 4, 8.9)
        box_b = Box2D(1.1, 1.27, 3.4, 5)

        # Act
        result = iou_2d(box_a, box_b)

        # Assert
        self.assertAlmostEqual(
            result,
            (3.4 - 2.3) * (5 - 3.7) / ((4 - 2.3) * (8.9 - 3.7) + (3.4 - 1.1) * (5 - 1.27) - (3.4 - 2.3) * (5 - 3.7)),
            places=8
        )

    def test_box_a_bottom_right_relative_box_b_case_3(self):
        # Arrange
        box_a = Box2D(1, 1, 6, 8)
        box_b = Box2D(1, 1, 3, 3.5)

        # Act
        result = iou_2d(box_a, box_b)

        # Assert
        self.assertAlmostEqual(result, 2 * 2.5 / (5 * 7 + 2 * 2.5 - 2 * 2.5), places=8)
