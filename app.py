#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
from collections import Counter
from collections import deque

import cv2 as cv

from application.application_mode import select_mode, ApplicationMode
from infrastructure.argument_parser import get_arguments
from infrastructure.openCV.Keys import get_key_press
from infrastructure.openCV.draw.draw_overlays import draw_overlays_with_landmarks, draw_overlays
from domain.landmark_processor import calculate_landmark_list, pre_process_landmark, pre_process_point_history
from infrastructure.mediapipe.hands_initializer import initialize_mediapipe_hands, process_image
from infrastructure.openCV.video_capture.video_capture_lifecycle import initialize_video_capture, read_image, \
    flip, correct_color
from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier
from infrastructure.model.data_source.csv_client import log_key_points_to_csv, \
    log_point_history_to_csv


def main():
    # Argument parsing #################################################################
    arguments = get_arguments()

    # Camera preparation ###############################################################
    capture = initialize_video_capture(arguments)

    # Model load #############################################################
    hands = initialize_mediapipe_hands(arguments)

    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier()

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################
    mode = ApplicationMode.PLAY

    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
        key = get_key_press()
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture #####################################################
        ret, image = read_image(capture)
        if not ret:
            break
        flipped_image = flip(image)  # Mirror display
        debug_image = copy.deepcopy(flipped_image)

        # Detection implementation #############################################################
        color_corrected_image = correct_color(flipped_image)

        color_corrected_image.lock()
        results = process_image(hands, color_corrected_image)
        color_corrected_image.unlock()

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            hand_sign_id, handedness, landmark_list, most_common_finger_gestures, hand_landmarks = process_landmarks(
                debug_image, finger_gesture_history,
                keypoint_classifier,
                point_history,
                point_history_classifier, mode, number, results)
            debug_image_with_landmark = draw_overlays_with_landmarks(debug_image, hand_sign_id, handedness,
                                                                     landmark_list,
                                                                     most_common_finger_gestures,
                                                                     hand_landmarks)
            debug_image_with_landmark_overlays = draw_overlays(debug_image_with_landmark, fps, mode, number,
                                                               point_history)
            cv.imshow('Hand Gesture Recognition', debug_image_with_landmark_overlays.image)

        else:
            point_history.append([0, 0])
            debug_image_with_overlays = draw_overlays(debug_image, fps, mode, number, point_history)
            cv.imshow('Hand Gesture Recognition', debug_image_with_overlays.image)

    capture.release()
    cv.destroyAllWindows()


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
                hand_sign_id = read_gesture(finger_gesture_history, keypoint_classifier, landmark_list,
                                            point_history, point_history_classifier, point_history_list,
                                            pre_processed_landmark_list)
            case ApplicationMode.LEARN_POINT_HISTORY:
                log_point_history(number, point_history_list)
                hand_sign_id = read_gesture(finger_gesture_history, keypoint_classifier, landmark_list,
                                            point_history, point_history_classifier, point_history_list,
                                            pre_processed_landmark_list)
            case ApplicationMode.LEARN_KEY_POINTS:
                log_key_points(number, pre_processed_landmark_list)
                hand_sign_id = read_gesture(finger_gesture_history, keypoint_classifier, landmark_list,
                                            point_history, point_history_classifier, point_history_list,
                                            pre_processed_landmark_list)

        return hand_sign_id, handedness, landmark_list, Counter(
            finger_gesture_history).most_common(), hand_landmarks


def read_gesture(finger_gesture_history, keypoint_classifier, landmark_list, point_history, point_history_classifier,
                 point_history_list, pre_processed_landmark_list):
    # should happen only if application is play
    # Hand sign classification
    hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
    if hand_sign_id == 2:  # Point gesture
        point_history.append(landmark_list[8])
    else:
        point_history.append([0, 0])
        # I can get landmark id from here and calculate absolute distance from individual landmarks
        # they can be mapped to midi messages, values etc.

        # Finger gesture classification
    finger_gesture_id = 0
    point_history_len = len(point_history_list)
    if point_history_len == (point_history.maxlen.real * 2):
        finger_gesture_id = point_history_classifier(
            point_history_list)

        # Calculates the gesture IDs in the latest detection
    finger_gesture_history.append(finger_gesture_id)
    return hand_sign_id


def play():
    print("todo")


def log_key_points(number, landmark_list):
    log_key_points_to_csv(number, landmark_list)


def log_point_history(number, point_history_list):
    log_point_history_to_csv(number, point_history_list)


if __name__ == '__main__':
    main()
