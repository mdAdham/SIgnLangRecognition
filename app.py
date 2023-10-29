#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dataclasses import dataclass

from application.application_mode import select_mode
from application.initialize_application import initialize_application
from domain.Labels import KeyPointLabel, PointHistoryLabel
from domain.gesture_reader import read_gesture
from infrastructure.mediapipe.process_image import process_image
from infrastructure.openCV.Keys import get_key_press
from domain.draw_overlays import draw_overlays_with_landmarks, draw_overlays
from domain.landmark_processor import calculate_landmark_list, pre_process_point_history, \
    pre_process_landmark, log_data
from infrastructure.openCV.video_capture.video_capture_lifecycle import read_image, show_frame, destroy_windows
from collections import Counter


def main():
    # Argument parsing #################################################################
    capture, cv_fps_calc, finger_gesture_history, \
        hands, keypoint_classifier, mode, point_history, point_history_classifier = initialize_application()

    while True:
        fps = cv_fps_calc.get()

        key = get_key_press()
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)
        ret, image = read_image(capture)
        if not ret:
            break
        processable_image, debug_image, ret = image.prepare()
        results = process_image(hands, processable_image)

        if results.multi_hand_landmarks is not None:

            hand_landmarks, handedness = zip(results.multi_hand_landmarks,
                                             results.multi_handedness)

            landmark_list = calculate_landmark_list(debug_image.image, hand_landmarks)

            pre_processed_landmark_list = pre_process_landmark(landmark_list)

            point_history_list = pre_process_point_history(debug_image.width(), debug_image.height(),
                                                           point_history)

            hand_sign: KeyPointLabel = read_gesture(finger_gesture_history, keypoint_classifier, landmark_list,
                                                    point_history, point_history_classifier, point_history_list,
                                                    pre_processed_landmark_list)
            finger_gesture = PointHistoryLabel(Counter(
                finger_gesture_history).most_common()[0][0]),
            debug_image_with_landmark = draw_overlays_with_landmarks(debug_image, hand_sign, handedness,
                                                                     landmark_list,
                                                                     finger_gesture,
                                                                     hand_landmarks)
            debug_image_with_landmark_overlays = draw_overlays(debug_image_with_landmark, fps, mode, number,
                                                               point_history)
            show_frame(debug_image_with_landmark_overlays)
            log_data(mode, number, point_history_list, pre_processed_landmark_list)
        else:
            point_history.append([0, 0])
            debug_image_with_overlays = draw_overlays(debug_image, fps, mode, number, point_history)
            show_frame(debug_image_with_overlays)

    capture.release()
    destroy_windows()


if __name__ == '__main__':
    main()
