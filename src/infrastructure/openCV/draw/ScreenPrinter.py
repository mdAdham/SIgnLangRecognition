import numpy as np
from src.application.application_mode import ApplicationMode
from src.application.opencv.Image import Image
from src.domain.Hands import Hand, Knuckle
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
        #if hand is not None:
            # rectangle = self.calculate_bounding_rectangle(image, hand)
            # self.draw_rectangle(image, rectangle)
            # self.draw_info_text(image, hand, rectangle, hand_sign, finger_gesture)
        self.draw_point_history(image, point_history)
        self.draw_statistics(image, fps, mode, number)
        self.show_frame(image)

    def draw_hand(self, image: Image, hand: Hand):
        draw_landmarks(image, hand)

    def draw_rectangle(self, image: Image, bounding_rectangle: list):
        cv.rectangle(image.image, (bounding_rectangle[0], bounding_rectangle[1]),
                     (bounding_rectangle[2], bounding_rectangle[3]),
                     (0, 0, 0), 1)

    def calculate_bounding_rectangle(self, image: Image, hand: Hand):
        landmark_array = np.empty((0, 2), int)

        for knuckle in hand.knuckles:
            landmark_x = min(int(knuckle.x * image.width()), image.width() - 1)
            landmark_y = min(int(knuckle.y * image.height()), image.height() - 1)

            landmark_point = [np.array((landmark_x, landmark_y))]

            landmark_array = np.append(landmark_array, landmark_point, axis=0)

        x, y, w, h = cv.boundingRect(landmark_array)

        return [x, y, x + w, y + h]

    def draw_info_text(self, image: Image, hand: Hand, bounding_rectangle, hand_sign: HandSignLabel,
                       finger_gesture: FingerGestureLabel):

        cv.putText(image.image, "".join([hand.handedness, ':', hand_sign]),
                   (bounding_rectangle[0] + 5, bounding_rectangle[1] - 4),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

        if finger_gesture is not None:
            cv.putText(image.image, "".join(["Finger Gesture:", finger_gesture]), (10, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
            cv.putText(image.image, "".join(["Finger Gesture:", finger_gesture]), (10, 60),
                       cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                       cv.LINE_AA)

        return image

    def draw_point_history(self, image: Image, point_history):
        for index, point in enumerate(point_history):
            if point[0] != 0 and point[1] != 0:
                cv.circle(image.image, point, 1 + int(index / 2),
                          (152, 251, 152), 2)

        return image

    def draw_statistics(self, image: Image, frames_per_second: int, mode: ApplicationMode, number: int):
        cv.putText(image.image, "FPS:" + str(frames_per_second), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4,
                   cv.LINE_AA)
        cv.putText(image.image, "FPS:" + str(frames_per_second), (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1.0,
                   (255, 255, 255), 2,
                   cv.LINE_AA)

        if mode != ApplicationMode.PLAY:
            cv.putText(image.image, "MODE:" + mode.name, (10, 90),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image.image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
        return image

    def show_frame(self, image: Image):
        cv.imshow('Hand Gesture Recognition', image.image)
