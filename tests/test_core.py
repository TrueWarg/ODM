from object_detection_metrics.core import (
    iou_2d, precision_binary, precision_micro, recall_binary, recall_micro
)
import unittest
import numpy as np


class IoU2DTest(unittest.TestCase):
    def test_box_a_top_left_relative_box_b_case_1(self):
        # Arrange
        box_a = np.array([0, 0, 4, 4])
        box_b = np.array([2, 2, 6, 6])

        # Act
        result = iou_2d(box_a, box_b)

        # Assert
        self.assertAlmostEqual(result, 2 * 2 / (4 * 4 + 4 * 4 - 2 * 2), places=8)

    def test_box_a_top_left_relative_box_b_case_2(self):
        # Arrange
        box_a = np.array([20, 20, 80, 80])
        box_b = np.array([40, 40, 90, 90])

        # Act
        result = iou_2d(box_a, box_b)

        # Assert
        self.assertAlmostEqual(result, 40 * 40 / (60 * 60 + 50 * 50 - 40 * 40), places=8)

    def test_box_a_top_left_relative_box_b_case_3(self):
        # Arrange
        box_a = np.array([3, 7, 8, 9])
        box_b = np.array([3, 7, 8, 10])

        # Act
        result = iou_2d(box_a, box_b)

        # Assert
        self.assertAlmostEqual(result, 5 * 2 / (5 * 2 + 5 * 3 - 5 * 2), places=8)

    def test_box_a_top_right_relative_box_b_case_1(self):
        # Arrange
        box_a = np.array([400, 30.5, 420, 50])
        box_b = np.array([350, 35.7, 410, 52.01])

        # Act
        result = iou_2d(box_a, box_b)

        # Assert
        self.assertAlmostEqual(result, 10 * 14.3 / (20 * 19.5 + 60 * 16.31 - 10 * 14.3), places=8)

    def test_box_a_top_right_relative_box_b_case_2(self):
        # Arrange
        box_a = np.array([20, 12, 36, 13.5])
        box_b = np.array([20, 12.5, 36, 14])

        # Act
        result = iou_2d(box_a, box_b)

        # Assert
        self.assertAlmostEqual(result, 16 * 1 / (16 * 1.5 + 16 * 1.5 - 16 * 1), places=8)

    def test_box_a_top_right_relative_box_b_case_3(self):
        # Arrange
        box_a = np.array([10, 10, 20, 20])
        box_b = np.array([5, 20, 20, 30])

        # Act
        result = iou_2d(box_a, box_b)

        # Assert
        self.assertAlmostEqual(result, 0, places=8)

    def test_box_a_bottom_right_relative_box_b_case_1(self):
        # Arrange
        box_a = np.array([4, 4, 10, 10])
        box_b = np.array([0, 0, 5, 5])

        # Act
        result = iou_2d(box_a, box_b)

        # Assert
        self.assertAlmostEqual(result, 1 * 1 / (6 * 6 + 5 * 5 - 1 * 1), places=8)

    def test_box_a_bottom_right_relative_box_b_case_2(self):
        # Arrange
        box_a = np.array([2.3, 3.7, 4, 8.9])
        box_b = np.array([1.1, 1.27, 3.4, 5])

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
        box_a = np.array([1, 1, 6, 8])
        box_b = np.array([1, 1, 3, 3.5])

        # Act
        result = iou_2d(box_a, box_b)

        # Assert
        self.assertAlmostEqual(result, 2 * 2.5 / (5 * 7 + 2 * 2.5 - 2 * 2.5), places=8)

    def test_box_a_bottom_left_relative_box_b_case_1(self):
        # Arrange
        box_a = np.array([0, 1, 3, 3.2])
        box_b = np.array([2.4, 0, 4, 3])

        # Act
        result = iou_2d(box_a, box_b)

        # Assert
        self.assertAlmostEqual(
            result,
            (3 - 2.4) * (3 - 1) / ((3 - 0) * (3.2 - 1) + (4 - 2.4) * (3 - 0) - (3 - 2.4) * (3 - 1)),
            places=8
        )

    def test_box_a_bottom_left_relative_box_b_case_2(self):
        # Arrange
        box_a = np.array([2, 2, 4, 7])
        box_b = np.array([3, 2, 6, 7])

        # Act
        result = iou_2d(box_a, box_b)

        # Assert
        self.assertAlmostEqual(result, (4 - 3) * (7 - 2) / (2 * 5 + 3 * 5 - 1 * 5), places=8)

    def test_box_a_bottom_left_relative_box_b_case_3(self):
        # Arrange
        box_a = np.array([2, 2, 4, 7])
        box_b = np.array([4, 2, 9, 7])

        # Act
        result = iou_2d(box_a, box_b)

        # Assert
        self.assertAlmostEqual(result, 0, places=8)


class PrecisionBinaryTest(unittest.TestCase):
    def test_case_1(self):
        # Arrange
        ground_truth = np.array([0, 0, 0, 1, 1])
        predicted = np.array([1, 0, 0, 1, 1])

        # Act
        result = precision_binary(ground_truth, predicted)

        # Assert
        self.assertAlmostEqual(result, 2 / (2 + 1), places=8)

    def test_case_2(self):
        # Arrange
        ground_truth = np.array([1, 0, 0, 1, 1, 1, 0])
        predicted = np.array([1, 0, 0, 1, 1, 0, 0])

        # Act
        result = precision_binary(ground_truth, predicted)

        # Assert
        self.assertAlmostEqual(result, 3 / (3 + 0), places=8)

    def test_case_3(self):
        # Arrange
        ground_truth = np.array([1, 1, 0, 1, 1, 1, 0])
        predicted = np.array([1, 0, 1, 1, 1, 0, 1])

        # Act
        result = precision_binary(ground_truth, predicted)

        # Assert
        self.assertAlmostEqual(result, 3 / (3 + 2), places=8)

    def test_case_4_all_zeros(self):
        # Arrange
        ground_truth = np.array([0, 0, 0, 0, 0, 0, 0])
        predicted = np.array([0, 0, 0, 0, 0, 0, 0])

        # Act
        result = precision_binary(ground_truth, predicted)

        # Assert
        self.assertAlmostEqual(result, 0.0, places=8)


class PrecisionMicroTest(unittest.TestCase):
    def test_3_classes_case_1(self):
        # Arrange
        ground_truth = np.array([0, 1, 2, 0, 2, 0, 1])
        predicted = np.array([0, 1, 2, 0, 2, 2, 0])

        # Act
        result = precision_micro(ground_truth, predicted)

        # Assert
        self.assertAlmostEqual(result, (2 + 1 + 2) / (2 + 1 + 2 + 1 + 0 + 1), places=8)

    def test_3_classes_case_2(self):
        # Arrange
        ground_truth = np.array([0, 1, 2, 0, 2, 0, 1, 0, 1])
        predicted = np.array([2, 1, 2, 1, 2, 2, 1, 1, 1])

        # Act
        result = precision_micro(ground_truth, predicted)

        # Assert
        self.assertAlmostEqual(result, (0 + 3 + 2) / (0 + 3 + 2 + 0 + 2 + 2), places=8)

    def test_4_classes_case_1(self):
        # Arrange
        ground_truth = np.array([0, 1, 2, 3, 2, 0, 0, 3, 1, 3])
        predicted = np.array([2, 1, 2, 3, 2, 2, 0, 1, 1, 3])

        # Act
        result = precision_micro(ground_truth, predicted)

        # Assert
        self.assertAlmostEqual(result, (1 + 2 + 2 + 2) / (1 + 2 + 2 + 2 + 0 + 1 + 2 + 0), places=8)


class RecallBinaryTest(unittest.TestCase):
    def test_case_1(self):
        # Arrange
        ground_truth = np.array([1, 0, 0, 1, 1])
        predicted = np.array([1, 0, 0, 1, 1])

        # Act
        result = recall_binary(ground_truth, predicted)

        # Assert
        self.assertAlmostEqual(result, 1, places=8)

    def test_case_2_all_zeros(self):
        # Arrange
        ground_truth = np.array([0, 0, 0])
        predicted = np.array([0, 0, 0])

        # Act
        result = recall_binary(ground_truth, predicted)

        # Assert
        self.assertAlmostEqual(result, 0, places=8)

    def test_case_3(self):
        # Arrange
        ground_truth = np.array([1, 1, 0, 1, 1])
        predicted = np.array([1, 0, 1, 1, 1])

        # Act
        result = recall_binary(ground_truth, predicted)

        # Assert
        self.assertAlmostEqual(result, 3 / (3 + 1), places=8)

    def test_case_4(self):
        # Arrange
        ground_truth = np.array([1, 1, 1, 1, 1, 0, 0, 1])
        predicted = np.array([1, 0, 1, 1, 1, 0, 1, 0])

        # Act
        result = recall_binary(ground_truth, predicted)

        # Assert
        self.assertAlmostEqual(result, 4 / (4 + 2), places=8)

    def test_case_5(self):
        # Arrange
        ground_truth = np.array([1, 1, 0, 1, 1])
        predicted = np.array([1, 0, 1, 0, 0])

        # Act
        result = recall_binary(ground_truth, predicted)

        # Assert
        self.assertAlmostEqual(result, 1 / (1 + 3), places=8)


class RecallMicroTest(unittest.TestCase):
    def test_3_classes_case_1(self):
        # Arrange
        ground_truth = np.array([0, 1, 2, 0, 2, 0, 1])
        predicted = np.array([0, 1, 2, 0, 2, 2, 0])

        # Act
        result = recall_micro(ground_truth, predicted)

        # Assert
        self.assertAlmostEqual(result, (2 + 1 + 2) / (2 + 1 + 2 + 1 + 1 + 0), places=8)

    def test_3_classes_case_2(self):
        # Arrange
        ground_truth = np.array([0, 1, 2, 0, 2, 0, 1, 2, 2])
        predicted = np.array([2, 1, 2, 1, 2, 2, 1, 1, 1])

        # Act
        result = recall_micro(ground_truth, predicted)

        # Assert
        self.assertAlmostEqual(result, (0 + 2 + 2) / (0 + 2 + 2 + 3 + 0 + 2), places=8)

    def test_4_classes_case_1(self):
        # Arrange
        ground_truth = np.array([0, 1, 2, 3, 2, 0, 0])
        predicted = np.array([2, 1, 2, 3, 2, 2, 0])

        # Act
        result = recall_micro(ground_truth, predicted)

        # Assert
        self.assertAlmostEqual(result, (1 + 1 + 2 + 1) / (1 + 1 + 2 + 1 + 2 + 0 + 0 + 0), places=8)