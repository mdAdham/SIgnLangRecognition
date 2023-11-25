import pickle
import unittest
import cv2 as cv

from src.application.opencv.Image import Image
from src.infrastructure.mediapipe.HandsReader import HandsReader
from test.HandsStub import should_create_hands


class HandsReaderTest(unittest.TestCase):
    use_static_image_mode = True
    min_detection_confidence = 0.5
    min_tracking_confidence = 0.5

    hands_reader = HandsReader(
        use_static_image_mode,
        min_detection_confidence,
        min_tracking_confidence,
    )

    def test_should_read_hand(self):
        test_image = Image(cv.imread('open hand.jpg'))
        expected_hands = should_create_hands()

        hands = self.hands_reader.get_hands(test_image)

        self.assertEqual(len(hands.hands_list[0].knuckles), 21)
        self.assertEqual(hands.hands_list[0].handedness, expected_hands.hands_list[0].handedness)
        self.assertEqual(hands.hands_list[0].get_base(), expected_hands.hands_list[0].get_base())
        self.assertEqual(hands.hands_list[0].get_index(), expected_hands.hands_list[0].get_index())


if __name__ == '__main__':
    unittest.main()
