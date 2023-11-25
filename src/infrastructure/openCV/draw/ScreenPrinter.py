import numpy as np
from src.application.application_mode import ApplicationMode
from src.application.opencv.Image import Image
from src.domain.Hands import Hand
from src.domain.Labels import FingerGestureLabel, HandSignLabel
from src.infrastructure.openCV.draw.draw_hand_landmarks import draw_landmarks
import cv2 as cv


class ScreenPrinter:

    def print_screen(
            self,
            image: Image,
            point_history,
            mode: ApplicationMode,
            fps: int,
            number: int,
            hand: Hand = None,
            hand_sign: HandSignLabel = None,
            finger_gesture: FingerGestureLabel = None,
    ):
        image_array = image.image
        if hand is not None:
            rectangle = self.calculate_bounding_rectangle(hand)
            image_array = self.draw_rectangle(image_array, rectangle)
            image_array = self.draw_hand(image_array, hand)
            image_array = self.draw_info_text(image_array, hand, rectangle, hand_sign, finger_gesture)
        image_array = self.draw_point_history(image_array, point_history)
        image_array = self.draw_statistics(image_array, fps, mode, number)
        self.show_frame(image_array)

    def draw_hand(self, image: np.ndarray, hand: Hand):
        return draw_landmarks(image, hand)

    def draw_rectangle(self, image: np.ndarray, bounding_rectangle: list):
        return cv.rectangle(image, (bounding_rectangle[0], bounding_rectangle[1]),
                            (bounding_rectangle[2], bounding_rectangle[3]),
                            (0, 0, 0), 1)

    def calculate_bounding_rectangle(self, hand: Hand):
        landmark_array = np.empty((0, 2), int)

        for knuckle in hand.knuckles:
            landmark_array = np.append(landmark_array, [np.array((knuckle.x, knuckle.y))], axis=0)

        x, y, w, h = cv.boundingRect(landmark_array)

        return [x, y, x + w, y + h]

    def draw_info_text(self, image: np.ndarray, hand: Hand, bounding_rectangle, hand_sign: HandSignLabel,
                       finger_gesture: FingerGestureLabel):

        cv.putText(image, "".join([hand.handedness.name, ':', hand_sign.name]),
                   (bounding_rectangle[0] + 5, bounding_rectangle[1] - 4),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

        if finger_gesture is not None:
            cv.putText(image, "".join(["Finger Gesture:", finger_gesture.name]), (10, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
            cv.putText(image, "".join(["Finger Gesture:", finger_gesture.name]), (10, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                       cv.LINE_AA)

        return image

    def draw_point_history(self, image: np.ndarray, point_history):
        for index, point in enumerate(point_history):
            if point[0] != 0 and point[1] != 0:
                cv.circle(image, point, 1 + int(index / 2),
                          (152, 251, 152), 2)

        return image

    def draw_statistics(self, image: np.ndarray, frames_per_second: int, mode: ApplicationMode, number: int):
        cv.putText(image, "FPS:" + str(frames_per_second), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4,
                   cv.LINE_AA)
        cv.putText(image, "FPS:" + str(frames_per_second), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0,
                   (255, 255, 255), 2,
                   cv.LINE_AA)

        if mode != ApplicationMode.PLAY:
            cv.putText(image, "MODE:" + mode.name, (10, 90),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
        return image

    def show_frame(self, image: np.ndarray):
        cv.imshow('Hand Gesture Recognition', image)
