import unittest
import cv2 as cv

from src.application.opencv.Image import Image
from src.infrastructure.mediapipe.HandsReader import HandsReader


class HandsReaderTest(unittest.TestCase):

    def test_should_read_hand(self):
        test_image = Image(cv.imread('open hand.jpg'))

        use_static_image_mode = True
        min_detection_confidence = 0.5
        min_tracking_confidence = 0.5
        key_point_model_path = '../src/model/keypoint_classifier/keypoint_classifier.tflite'
        point_history_model_path = '../src/model/point_history_classifier/point_history_classifier.tflite'
        hands_reader = HandsReader(
            use_static_image_mode,
            min_detection_confidence,
            min_tracking_confidence,
            key_point_model_path,
            point_history_model_path,
        )

        hands = hands_reader.process_image(test_image)

        self.assertEqual(len(hands.hands_list[0].landmarks), 21)

if __name__ == '__main__':
    unittest.main()