import copy
import itertools
from collections import Counter

from application.application_mode import ApplicationMode
from application.keypoint_logger import log_point_history, log_key_points
from domain.Labels import KeyPointLabel, PointHistoryLabel
from domain.gesture_reader import read_gesture


def calculate_landmark_list(image, landmarks) -> list:
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point


def pre_process_landmark(landmark_list: list) -> list:
    temp_landmark_list = copy.deepcopy(landmark_list)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, landmark_point in enumerate(temp_landmark_list):
        if index == 0:
            base_x, base_y = landmark_point[0], landmark_point[1]

        temp_landmark_list[index][0] = temp_landmark_list[index][0] - base_x
        temp_landmark_list[index][1] = temp_landmark_list[index][1] - base_y

    # Convert to a one-dimensional list
    temp_landmark_list = list(
        itertools.chain.from_iterable(temp_landmark_list))

    # Normalization
    max_value = max(list(map(abs, temp_landmark_list)))

    def normalize_(n):
        return n / max_value

    temp_landmark_list = list(map(normalize_, temp_landmark_list))

    return temp_landmark_list


def pre_process_point_history(image_width, image_height, point_history):
    temp_point_history = copy.deepcopy(point_history)

    # Convert to relative coordinates
    base_x, base_y = 0, 0
    for index, point in enumerate(temp_point_history):
        if index == 0:
            base_x, base_y = point[0], point[1]

        temp_point_history[index][0] = (temp_point_history[index][0] -
                                        base_x) / image_width
        temp_point_history[index][1] = (temp_point_history[index][1] -
                                        base_y) / image_height

    # Convert to a one-dimensional list
    temp_point_history = list(
        itertools.chain.from_iterable(temp_point_history))

    return temp_point_history


def process_landmarks(debug_image, finger_gesture_history, keypoint_classifier,
                      point_history, point_history_classifier, mode, number, results):
    for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                          results.multi_handedness):
        # Landmark calculation
        landmark_list = calculate_landmark_list(debug_image.image, hand_landmarks)

        # Conversion to relative coordinates / normalized coordinates
        pre_processed_landmark_list = pre_process_landmark(landmark_list)

        point_history_list = pre_process_point_history(debug_image.width(), debug_image.height(),
                                                       point_history)
        match mode:
            case ApplicationMode.PLAY:
                hand_sign: KeyPointLabel = read_gesture(finger_gesture_history, keypoint_classifier, landmark_list,
                                                        point_history, point_history_classifier, point_history_list,
                                                        pre_processed_landmark_list)
            case ApplicationMode.LEARN_POINT_HISTORY:
                log_point_history(number, point_history_list)
                hand_sign: KeyPointLabel = read_gesture(finger_gesture_history, keypoint_classifier, landmark_list,
                                                        point_history, point_history_classifier, point_history_list,
                                                        pre_processed_landmark_list)
            case ApplicationMode.LEARN_KEY_POINTS:
                log_key_points(number, pre_processed_landmark_list)
                hand_sign: KeyPointLabel = read_gesture(finger_gesture_history, keypoint_classifier, landmark_list,
                                                        point_history, point_history_classifier, point_history_list,
                                                        pre_processed_landmark_list)

        return hand_sign, handedness, landmark_list, PointHistoryLabel(Counter(
            finger_gesture_history).most_common()[0][0]), hand_landmarks
