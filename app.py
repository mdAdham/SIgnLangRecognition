#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy

from application.application_mode import select_mode
from application.initialize_application import initialize_application
from infrastructure.mediapipe.process_image import process_image
from infrastructure.openCV.Keys import get_key_press
from domain.draw_overlays import draw_overlays_with_landmarks, draw_overlays
from domain.landmark_processor import process_landmarks
from infrastructure.openCV.video_capture.video_capture_lifecycle import read_image, \
    flip, correct_color, show_frame, destroy_windows


def main():
    # Argument parsing #################################################################
    capture, cvFpsCalc, finger_gesture_history, hands, keypoint_classifier, mode, point_history, point_history_classifier = initialize_application()

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
            show_frame(debug_image_with_landmark_overlays)

        else:
            point_history.append([0, 0])
            debug_image_with_overlays = draw_overlays(debug_image, fps, mode, number, point_history)
            show_frame(debug_image_with_overlays)

    capture.release()
    destroy_windows()


if __name__ == '__main__':
    main()
