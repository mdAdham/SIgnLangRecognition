from src.application.application_mode import ApplicationMode
import numpy as np
import cv2 as cv

from src.infrastructure.openCV.video_capture.VideoCaptor import Image


def draw_bounding_rectangle(image, bounding_rectangle):
    cv.rectangle(image, (bounding_rectangle[0], bounding_rectangle[1]),
                 (bounding_rectangle[2], bounding_rectangle[3]),
                 (0, 0, 0), 1)

    return image


def calculate_bounding_rectangle(image: Image, landmarks):
    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image.width()), image.width() - 1)
        landmark_y = min(int(landmark.y * image.height()), image.height() - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]


def draw_info_text(image: cv.typing.MatLike, bounding_rectangle, chirality, hand_sign_text: str,
                   finger_gesture_text: str):
    cv.rectangle(image, (bounding_rectangle[0], bounding_rectangle[1]),
                 (bounding_rectangle[2], bounding_rectangle[1] - 22),
                 (0, 0, 0), -1)

    cv.putText(image, "".join([chirality, ':', hand_sign_text]), (bounding_rectangle[0] + 5, bounding_rectangle[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)

    if finger_gesture_text != "":
        cv.putText(image, "".join(["Finger Gesture:", finger_gesture_text]), (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(image, "".join(["Finger Gesture:" + finger_gesture_text]), (10, 60),
                   cv.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2,
                   cv.LINE_AA)

    return image


def draw_point_history(image: cv.typing.MatLike, point_history):
    for index, point in enumerate(point_history):
        if point[0] != 0 and point[1] != 0:
            cv.circle(image, (point[0], point[1]), 1 + int(index / 2),
                      (152, 251, 152), 2)

    return image


def draw_statistics(image: cv.typing.MatLike, fps, mode, number):
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (0, 0, 0), 4, cv.LINE_AA)
    cv.putText(image, "FPS:" + str(fps), (10, 30), cv.FONT_HERSHEY_SIMPLEX,
               1.0, (255, 255, 255), 2, cv.LINE_AA)

    if mode != ApplicationMode.PLAY:
        cv.putText(image, "MODE:" + mode.name, (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image
