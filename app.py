#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
from collections import Counter
from collections import deque

import cv2 as cv
from attr import dataclass

from application.application_mode import application_mode, select_mode
from infrastructure.argument_parser import get_arguments
from infrastructure.openCV.draw.draw_overlays import draw_overlays_with_landmarks, draw_overlays
from domain.landmark_processor import calc_landmark_list, pre_process_landmark, pre_process_point_history
from infrastructure.mediapipe.hands_initializer import initialize_mediapipe_hands
from infrastructure.openCV.video_capture.video_capture_initializer import initialize_video_capture
from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier
from infrastructure.model_data_source.csv_client import log_to_csv, read_from_csv


def main():
    # Argument parsing #################################################################
    arguments = get_arguments()

    use_bounding_rectangle = True

    # Camera preparation ###############################################################
    capture = initialize_video_capture(cv, arguments)

    # Model load #############################################################
    hands = initialize_mediapipe_hands(arguments)

    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier()

    # Read labels ###########################################################
    keypoint_classifier_labels, point_history_classifier_labels = read_from_csv()

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # Coordinate history #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # Finger gesture history ################################################
    finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################
    mode = application_mode['PLAY']

    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture #####################################################
        ret, image = capture.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True

        #  ####################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks,
                                                  results.multi_handedness):
                # Landmark calculation
                landmark_list = calc_landmark_list(debug_image, hand_landmarks)

                # Conversion to relative coordinates / normalized coordinates
                pre_processed_landmark_list = pre_process_landmark(
                    landmark_list)
                pre_processed_point_history_list = pre_process_point_history(
                    debug_image, point_history)
                # Write to the dataset file
                log_to_csv(number, mode, pre_processed_landmark_list,
                           pre_processed_point_history_list)

                # Hand sign classification
                hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                if hand_sign_id == 2:  # Point gesture
                    point_history.append(landmark_list[8])
                else:
                    point_history.append([0, 0])
                # Send Midi here
                # I can get landmark id from here and calculate absolute distance from individual landmarks
                # they can be mapped to midi messages, values etc.

                # Finger gesture classification
                finger_gesture_id = 0
                point_history_len = len(pre_processed_point_history_list)
                if point_history_len == (history_length * 2):
                    finger_gesture_id = point_history_classifier(
                        pre_processed_point_history_list)

                # Calculates the gesture IDs in the latest detection
                finger_gesture_history.append(finger_gesture_id)

                # Drawing part
                debug_image = draw_overlays_with_landmarks(debug_image, hand_sign_id, handedness,
                                                           keypoint_classifier_labels, landmark_list,
                                                           Counter(finger_gesture_history).most_common(),
                                                           point_history_classifier_labels, use_bounding_rectangle,
                                                           hand_landmarks, cv)
        else:
            point_history.append([0, 0])

        debug_image = draw_overlays(debug_image, fps, mode, number, point_history, cv)

        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

    capture.release()
    cv.destroyAllWindows()


@dataclass
class OverlayParams:
    image: cv.typing.MatLike
    cv: cv


if __name__ == '__main__':
    main()
