import itertools
from collections import deque

import mediapipe as mp

from src.application.opencv.Image import Image
from src.domain.Hands import Hands, Hand, Chirality, Knuckle


class HandsReader:
    max_num_hands = 2

    def __init__(
            self,
            use_static_image_mode: bool,
            min_detection_confidence: float,
            min_tracking_confidence: float,

    ):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=use_static_image_mode,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def scale_landmarks(self, image: Image, landmarks):
        image_width, image_height = image.image.shape[1], image.image.shape[0]

        landmark_point = []

        # Keypoint
        for _, landmark in enumerate(landmarks):
            landmark_x = min(int(landmark.x * image_width), image_width - 1)
            landmark_y = min(int(landmark.y * image_height), image_height - 1)
            # landmark_z = landmark.z

            landmark_point.append([landmark_x, landmark_y])

        return landmark_point

    def get_hands(self, image: Image) -> Hands | None:
        frame = self.hands.process(image.image)
        hands_list = []
        if frame.multi_hand_landmarks is None:
            return None
        for hand_landmarks, handednness in zip(frame.multi_hand_landmarks, frame.multi_handedness):
            hands_list.append(
                Hand(
                    list(map(
                        lambda n: Knuckle(x=n[0], y=n[1]),
                        self.scale_landmarks(image, hand_landmarks.landmark))),
                    Chirality(handednness.classification[0].label[0:])
                )
            )
        return Hands(hands_list)
