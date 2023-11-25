import unittest

from src.infrastructure.model.GestureReader import GestureReader


class GestureReaderTest(unittest.TestCase):
    key_point_model_path = '../src/model/keypoint_classifier/keypoint_classifier.tflite'
    point_history_model_path = '../src/model/point_history_classifier/point_history_classifier.tflite'
    gesture_reader = GestureReader(key_point_model_path,
                                   point_history_model_path)

    def test_should_read_gesture(self):
        self.assertEqual(True, False)  # add assertion here


if __name__ == '__main__':
    unittest.main()
