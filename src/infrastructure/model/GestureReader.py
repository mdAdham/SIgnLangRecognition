import itertools
from collections import deque, Counter

import numpy as np
from numpy import ndarray

from src.domain.Hands import Hand, Knuckle
from src.domain.Labels import HandSignLabel, FingerGestureLabel
from src.model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from src.model.point_history_classifier.point_history_classifier import PointHistoryClassifier


class GestureReader:
    history_length = 16

    point_history: deque = deque(maxlen=history_length)
    gesture_history: deque = deque(maxlen=history_length)

    def __init__(self,
                 key_point_model_path='src/model/keypoint_classifier/keypoint_classifier.tflite',
                 point_history_model_path='src/model/point_history_classifier/point_history_classifier.tflite',
                 ):
        self.key_point_classifier = KeyPointClassifier(model_path=key_point_model_path)
        self.point_history_classifier = PointHistoryClassifier(model_path=point_history_model_path)

    def read(self, hand: Hand) -> tuple[HandSignLabel, FingerGestureLabel]:
        hand_sign = self.read_hand_sign(hand)
        self.append_point_history(hand, hand_sign)
        finger_gesture = self.read_finger_gesture()
        return hand_sign, finger_gesture

    def read_hand_sign(self,
                       hand: Hand,
                       ) -> HandSignLabel:

        return self.key_point_classifier(hand.prepare_for_model())

    def append_point_history(self, hand, hand_sign):
        if hand_sign == HandSignLabel.POINTER:
            index = hand.get_index()
            self.point_history.append(hand.get_index())  # I suggest that this is magic number for index
        else:
            self.point_history.append(Knuckle(0.0, 0.0))

    def point_history_to_list(self) -> list:
        one_dimensional_point_history = []
        for index, knuckle in enumerate(self.point_history):
            one_dimensional_point_history[index][0] = knuckle.x
            one_dimensional_point_history[index][1] = knuckle.y
        return list(
        itertools.chain.from_iterable(one_dimensional_point_history))

    def read_finger_gesture(self) -> FingerGestureLabel:
        finger_gesture_id = 0
        point_history_len = len(self.point_history)
        if point_history_len == (self.point_history.maxlen.real * 2):
            finger_gesture_id = self.point_history_classifier(
                self.point_history_to_list())

        self.gesture_history.append(finger_gesture_id)
        return FingerGestureLabel(Counter(self.gesture_history).most_common()[0][0])
