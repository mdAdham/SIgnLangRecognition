from collections import deque, Counter

from src.domain.Hands import Hand
from src.domain.Labels import KeyPointLabel, PointHistoryLabel
from src.model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from src.model.point_history_classifier.point_history_classifier import PointHistoryClassifier


class GestureReader:
    history_length = 16

    point_history = deque(maxlen=history_length)
    gesture_history = deque(maxlen=history_length)

    def __init__(self,
                 key_point_model_path='src/model/keypoint_classifier/keypoint_classifier.tflite',
                 point_history_model_path='src/model/point_history_classifier/point_history_classifier.tflite',
                 ):
        self.key_point_classifier = KeyPointClassifier(model_path=key_point_model_path)
        self.point_history_classifier = PointHistoryClassifier(model_path=point_history_model_path)

    def read_hand_sign(self,
                       hand: Hand,
                       ) -> KeyPointLabel:

        hand_sign = self.key_point_classifier(hand.prepare_for_model())
        if hand_sign == KeyPointLabel.POINTER:
            self.point_history.append(hand.get_index())  # I suggest that this is magic number for index
        else:
            self.point_history.append([0, 0])

        return hand_sign
# todo: finger gesture relies on hand sign to be called before.
    def read_finger_gesture(self, ) -> PointHistoryLabel:
        finger_gesture_id = 0
        point_history_len = len(self.point_history)
        if point_history_len == (self.point_history.maxlen.real * 2):
            finger_gesture_id = self.point_history_classifier(
                self.point_history)

        self.gesture_history.append(finger_gesture_id)
        return PointHistoryLabel(Counter(self.point_history).most_common()[0][0])
