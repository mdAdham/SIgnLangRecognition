from collections import deque

import mediapipe as mp

from src.application.opencv.Image import Image
from src.domain.Hands import Hands, Hand, Chirality
from src.domain.Labels import KeyPointLabel
from src.model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from src.model.point_history_classifier.point_history_classifier import PointHistoryClassifier


class HandsReader:
    history_length = 16
    max_num_hands = 2
    point_history = deque(maxlen=history_length)
    gesture_history = deque(maxlen=history_length)

    def __init__(
            self,
            use_static_image_mode: bool,
            min_detection_confidence: float,
            min_tracking_confidence: float,
            key_point_model_path='src/model/keypoint_classifier/keypoint_classifier.tflite',
            point_history_model_path='src/model/point_history_classifier/point_history_classifier.tflite',

    ):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=use_static_image_mode,
            max_num_hands=self.max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.key_point_classifier = KeyPointClassifier(model_path=key_point_model_path)
        self.point_history_classifier = PointHistoryClassifier(model_path=point_history_model_path)

    def process_image(self, image: Image) -> Hands:
        frame = self.hands.process(image.image)
        hands_list = []
        for hand_landmarks, handednness in zip(frame.multi_hand_landmarks, frame.multi_handedness):
            hands_list.append(
                Hand(
                    hand_landmarks.landmark,
                    Chirality(handednness.classification[0].label[0:])
                )
            )
        return Hands(hands_list)

    def read_gesture(self, landmark_list, pre_processed_landmark_list, point_history_list) -> KeyPointLabel:

        hand_sign = self.key_point_classifier(pre_processed_landmark_list)
        if hand_sign == KeyPointLabel.POINTER:
            self.point_history.append(landmark_list[8])
        else:
            self.point_history.append([0, 0])

        finger_gesture_id = 0
        point_history_len = len(point_history_list)
        if point_history_len == (self.point_history.maxlen.real * 2):
            finger_gesture_id = self.point_history_classifier(
                point_history_list)

        self.gesture_history.append(finger_gesture_id)
        return hand_sign
