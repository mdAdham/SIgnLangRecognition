#!/usr/bin/env python
# -*- coding: utf-8 -*-

from src.application.application_mode import select_mode, ApplicationMode
from src.application.initialize_application import initialize_application
from src.domain.Hands import Knuckle
from src.domain.landmark_processor import log_data
from src.infrastructure.openCV.Keys import get_key_press


def main():
    # Argument parsing #################################################################
    video_capture, cv_fps_calc, \
        hands_reader, gesture_reader, screen_printer, mode = initialize_application()

    while True:
        fps = cv_fps_calc.get()

        key = get_key_press()
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)
        ret, image = video_capture.get_frame()
        if not ret:
            break
        processable_image, debug_image = image.prepare()
        processable_image.image.flags.writeable = False
        hands = hands_reader.get_hands(processable_image)
        processable_image.image.flags.writeable = True

        if hands is not None:

            for hand in hands.hands_list:
                hand_sign, finger_gesture = gesture_reader.read(hand)

                if mode != ApplicationMode.PLAY:
                    screen_printer.print_screen(
                        debug_image, gesture_reader.point_history, mode, fps, number, hand, hand_sign, finger_gesture)

                    log_data(mode, number, gesture_reader.point_history, hand.prepare_for_model())

                elif mode == ApplicationMode.PLAY:
                    print("todo: implement midi")
        else:
            gesture_reader.point_history.append(Knuckle(0, 0))
            if mode != ApplicationMode.PLAY:
                screen_printer.print_screen(
                    debug_image, gesture_reader.point_history, mode, fps, number)

    video_capture.destroy_windows()


if __name__ == '__main__':
    main()
