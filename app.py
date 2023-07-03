#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque
import random
import re
import time

import cv2 as cv
import numpy as np
import mediapipe as mp
from requests import post
import math

from utils import CvFpsCalc
from model import KeyPointClassifier


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=1)
    parser.add_argument("--width", help='cap width', type=int, default=960)
    parser.add_argument("--height", help='cap height', type=int, default=540)

    parser.add_argument('--use_static_image_mode', action='store_true')
    parser.add_argument("--min_detection_confidence",
                        help='min_detection_confidence',
                        type=float,
                        default=0.7)
    parser.add_argument("--min_tracking_confidence",
                        help='min_tracking_confidence',
                        type=int,
                        default=0.5)
    parser.add_argument("--max_num_hands", type=int, default=10)
    parser.add_argument("--use_brect", type=bool, default=True)
    parser.add_argument("--homeassistant_url", type=str, default='http://192.168.178.118:8123/')
    parser.add_argument("--homeassistant_header", type=dict, default={"Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJmZGU1YWQ5NjI1MDQ0NWE3YTI1NGRjYjM0NDAzYWU4MyIsImlhdCI6MTY4ODAzNjM0OSwiZXhwIjoyMDAzMzk2MzQ5fQ.s-GU1OqsLPW7HYo2ZkSGg6twERApfqIl7W0gjJraq20"})
    parser.add_argument("--smart_home_devices", type=list, default=["vacuum.ntelifamilyrobot", "light.deckenlampe", "light.haso_bett_led", "light.flur", "media_player.imrans_echo_dot", "select.siemens_ti9555x1de_68a40e325683_bsh_common_setting_powerstate"])

    args = parser.parse_args()

    return args


def main():
    # Argument parsing #################################################################
    args = get_args()
    cap_device = args.device
    cap_width = args.width
    cap_height = args.height
    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence
    max_num_hands = args.max_num_hands
    use_brect = args.use_brect
    smart_home_devices = args.smart_home_devices

    # Camera preparation ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # Model load #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=max_num_hands,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    # Read labels ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]

    # FPS Measurement ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    #  ########################################################################
    mode = 0

    smart_home_entity = "."
    duration_after_start = 6 # in seconds
    detected_hands = {}  # Dictionary to store the IDs of the hands showing the specific sign
   
    while True:
        fps = cvFpsCalc.get()

        # Process Key (ESC: end) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # Camera capture #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # Mirror display
        debug_image = copy.deepcopy(image)

        # Detection implementation #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False
        hand_results = hands.process(image)
        image.flags.writeable = True

        # Process detection hand_results #############################################################
        if hand_results.multi_hand_landmarks is not None:
            if len(detected_hands) <= 2:
                for hand_id, (hand_landmarks, handedness) in enumerate(
                    zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness)
                ):
                    pre_processed_landmark_list = pre_process_landmark(calc_landmark_list(debug_image, hand_landmarks))
                    hand_sign_class = keypoint_classifier_labels[keypoint_classifier(pre_processed_landmark_list)]
                    if hand_sign_class == "Shaka":
                        detected_hands[hand_id] = time.time() # Store the ID of the hand showing the "Shaka" sign
                        break

            if len(detected_hands) != 0:
                for hand_id, (hand_landmarks, handedness) in enumerate(
                    zip(hand_results.multi_hand_landmarks, hand_results.multi_handedness)
                ):
                    if hand_id in detected_hands and (time.time() - detected_hands[hand_id] <= duration_after_start):  # Process only the hands that showed the "Shaka" sign
                        # Which hand side
                        hand_side = handedness.classification[0].label[0:]
                        # Bounding box calculation
                        brect = calc_bounding_rect(debug_image, hand_landmarks)
                        # Landmark calculation
                        landmark_list = calc_landmark_list(debug_image, hand_landmarks)
                        # Conversion to relative coordinates / normalized coordinates
                        pre_processed_landmark_list = pre_process_landmark(landmark_list)
                        # Write to the dataset file
                        logging_csv(number, mode, pre_processed_landmark_list)
                        # Hand sign classification
                        hand_sign_id = keypoint_classifier(pre_processed_landmark_list)
                        hand_sign_class = keypoint_classifier_labels[hand_sign_id]
                        # Drawing part
                        debug_image = draw_bounding_rect(use_brect, debug_image, brect)
                        debug_image = draw_landmarks(debug_image, landmark_list)
                        debug_image = draw_info_text(
                            debug_image,
                            brect,
                            hand_side,
                            hand_sign_class,
                        )
                        if hand_sign_class == "Zero":
                            smart_home_entity = smart_home_devices[0]
                        elif hand_sign_class == "One":
                            smart_home_entity = smart_home_devices[1]
                        elif hand_sign_class == "Two":
                            smart_home_entity = smart_home_devices[2]
                        elif hand_sign_class == "Three":
                            smart_home_entity = smart_home_devices[3]
                        elif hand_sign_class == "Four":
                            smart_home_entity = smart_home_devices[4]
                        elif hand_sign_class == "Five":
                            smart_home_entity = smart_home_devices[5]
                        
                        custom_perfom_action(hand_sign_class, smart_home_entity, debug_image, landmark_list, brect)

                    elif hand_id in detected_hands and (time.time() - detected_hands[hand_id] > duration_after_start):
                        del detected_hands[hand_id]
            else:
                smart_home_entity = "."

        debug_image = draw_info(debug_image, fps, mode, number)
        # Screen reflection #############################################################
        cv.imshow('Hand Gesture Recognition', debug_image)

    cap.release()
    cv.destroyAllWindows()

def custom_perfom_action(hand_sign_class, smart_home_entity, debug_image, landmark_list, brect):
    args = get_args()
    homeassistant_url = args.homeassistant_url
    homeassistant_header = args.homeassistant_header
    smart_home_domain = re.match(r"^(.*?)\.", smart_home_entity).group(1)

    homeassistant_api_turnon = homeassistant_url+"api/services/homeassistant/turn_on"
    homeassistant_api_turnoff = homeassistant_url+"api/services/homeassistant/turn_off"
    homeassistant_api_playmedia = homeassistant_url+"api/services/media_player/play_media"
    if hand_sign_class == "ThumpUp":
        if smart_home_domain == "light" or smart_home_domain == "vacuum" or smart_home_domain == "select":
            data = {"entity_id": smart_home_entity}
            post(homeassistant_api_turnon, headers=homeassistant_header, json=data)
        elif smart_home_domain == "media_player":
            data = {"entity_id": smart_home_entity, "media_content_id": "Start", "media_content_type": "custom"}
            post(homeassistant_api_playmedia, headers=homeassistant_header, json=data)
    elif hand_sign_class == "ThumpDown":
        if smart_home_domain == "light" or smart_home_domain == "vacuum" or smart_home_domain == "select":
            data = {"entity_id": smart_home_entity}
            post(homeassistant_api_turnoff, headers=homeassistant_header, json=data)
        elif smart_home_domain == "media_player":
            data = {"entity_id": smart_home_entity, "media_content_id": "Stop", "media_content_type": "custom"}
            post(homeassistant_api_playmedia, headers=homeassistant_header, json=data)
    elif hand_sign_class == "Control":
        length, pointCoordinates = calc_finger_distance(landmark_list, brect, 4, 8) #thumbs to index finger
        debug_image = draw_distance(debug_image, length, pointCoordinates, [255,0,255], False)
        if not calc_finger_up(landmark_list, 18, 20): #little finger
            debug_image = draw_distance(debug_image, length, pointCoordinates, [0,255,0], True)
            if smart_home_domain == "light":
                brightness = int(np.interp(length, [0.15, 0.85], [1, 254])) # Hand range 0.15-0.85 || Brightness range 1-254
                data = {"entity_id": smart_home_entity, "brightness": brightness}
                post(homeassistant_api_turnon, headers=homeassistant_header, json=data)
            elif smart_home_domain == "media_player":
                volume = np.interp(length, [0.15, 0.85], [0.0, 1.0]) # Hand range 0.15-0.85 || Volume range 0.0 - 1.0
                api = homeassistant_url+"api/services/media_player/volume_set"
                data = {"entity_id": smart_home_entity, "volume_level": volume}
                post(api, headers=homeassistant_header, json=data)
    elif hand_sign_class == "Rock":
        if not calc_finger_up(landmark_list, 6, 8): #index finger
            if smart_home_domain == "light":
                random_rgb = [random.randint(0, 255) for _ in range(3)]
                data = {"entity_id": smart_home_entity, "rgb_color": random_rgb}
                post(homeassistant_api_turnon, headers=homeassistant_header, json=data)
            elif smart_home_domain == "media_player":
                data = {"entity_id": smart_home_entity, "media_content_id": "Next", "media_content_type": "custom"}
                post(homeassistant_api_playmedia, headers=homeassistant_header, json=data)
        elif not calc_finger_up(landmark_list, 18, 20): #little finger
            if smart_home_domain == "light":
                data = {"entity_id": smart_home_entity, "rgb_color": [255,255,255]}
                post(homeassistant_api_turnon, headers=homeassistant_header, json=data)
            elif smart_home_domain == "media_player":
                data = {"entity_id": smart_home_entity, "media_content_id": "Previous", "media_content_type": "custom"}
                post(homeassistant_api_playmedia, headers=homeassistant_header, json=data)

def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode

def calc_bounding_rect(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]

def calc_landmark_list(image, landmarks):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_point = []

    # Keypoint
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        # landmark_z = landmark.z

        landmark_point.append([landmark_x, landmark_y])

    return landmark_point

def calc_finger_distance(landmarks, brect, f1, f2):
    x1, y1 = landmarks[f1][0], landmarks[f1][1]
    x2, y2 = landmarks[f2][0], landmarks[f2][1]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    bbox_width = brect[2] - brect[0]
    bbox_height = brect[3] - brect[1]
    x1_norm = (landmarks[f1][0] - brect[0]) / bbox_width
    y1_norm = (landmarks[f1][1] - brect[1]) / bbox_height
    x2_norm = (landmarks[f2][0] - brect[0]) / bbox_width
    y2_norm = (landmarks[f2][1] - brect[1]) / bbox_height

    length = math.hypot(x2_norm - x1_norm, y2_norm - y1_norm)
    return length, [x1, y1, x2, y2, cx, cy]

def calc_finger_up(landmarks, f1, f2):
    y1, y2 = landmarks[f1][1], landmarks[f2][1]
    try:
        if y1 >= y2:
            return True
        elif y1 < y2:
            return False
    except:
        return "NO HAND FOUND"

def pre_process_landmark(landmark_list):
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


def logging_csv(number, mode, landmark_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = 'model/keypoint_classifier/keypoint.csv'
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    return

def draw_distance(image, length, pointCoordinates, rgb, isSet):
    if pointCoordinates[0]!=0 and pointCoordinates[1]!=0:
        cv.circle(image, (pointCoordinates[0], pointCoordinates[1]), 10, (rgb[0], rgb[1], rgb[2]), cv.FILLED)
    if pointCoordinates[2]!=0 and pointCoordinates[3]!=0:
        cv.circle(image, (pointCoordinates[2], pointCoordinates[3]), 10, (rgb[0], rgb[1], rgb[2]), cv.FILLED)
    if pointCoordinates[4]!=0 and pointCoordinates[5]!=0:
        cv.circle(image, (pointCoordinates[4], pointCoordinates[5]), 10, (rgb[0], rgb[1], rgb[2]), cv.FILLED)
    if pointCoordinates[0]!=0 and pointCoordinates[1]!=0 and pointCoordinates[2]!=0 and pointCoordinates[3]!=0:
        cv.line(image, (pointCoordinates[0], pointCoordinates[1]), (pointCoordinates[2], pointCoordinates[3]), (rgb[0], rgb[1], rgb[2]), 3)
    lengthBar = int(np.interp(length, [0.15, 0.85], [400,150]))
    smoothness = 5
    lengthPer = smoothness * round(int(np.interp(length, [0.15, 0.85], [0, 100]))/smoothness)
    if isSet == True:
        rgb = [0,255,0]
    else:
        rgb = [255,0,0]
    cv.rectangle(image, (50,150), (85,400), (rgb[0], rgb[1], rgb[2]), 3)
    cv.rectangle(image, (50, int(lengthBar)), (85,400), (rgb[0], rgb[1], rgb[2]), cv.FILLED)
    cv.putText(image, f'{int(lengthPer)}%', (40, 450), cv.FONT_HERSHEY_COMPLEX,
               1, (rgb[0], rgb[1], rgb[2]), 3)
    return image

def draw_landmarks(image, landmark_point):
    if len(landmark_point) > 0:
        # Thumb
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[3]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[3]), tuple(landmark_point[4]),
                (255, 255, 255), 2)

        # Index finger
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[6]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[6]), tuple(landmark_point[7]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[7]), tuple(landmark_point[8]),
                (255, 255, 255), 2)

        # Middle finger
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[10]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[10]), tuple(landmark_point[11]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[11]), tuple(landmark_point[12]),
                (255, 255, 255), 2)

        # Ring finger
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[14]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[14]), tuple(landmark_point[15]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[15]), tuple(landmark_point[16]),
                (255, 255, 255), 2)

        # Little finger
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[18]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[18]), tuple(landmark_point[19]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[19]), tuple(landmark_point[20]),
                (255, 255, 255), 2)

        # Palm
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[0]), tuple(landmark_point[1]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[1]), tuple(landmark_point[2]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[2]), tuple(landmark_point[5]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[5]), tuple(landmark_point[9]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[9]), tuple(landmark_point[13]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[13]), tuple(landmark_point[17]),
                (255, 255, 255), 2)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (0, 0, 0), 6)
        cv.line(image, tuple(landmark_point[17]), tuple(landmark_point[0]),
                (255, 255, 255), 2)

    # Key Points
    for index, landmark in enumerate(landmark_point):
        if index == 0:  # 手首1
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (landmark[0], landmark[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv.circle(image, (landmark[0], landmark[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (landmark[0], landmark[1]), 8, (0, 0, 0), 1)

    return image


def draw_bounding_rect(use_brect, image, brect):
    if use_brect:
        # Outer rectangle
        cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[3]),
                     (0, 0, 0), 2)

    return image


def draw_info_text(image, brect, handedness_side, hand_sign_text):
    cv.rectangle(image, (brect[0], brect[1]), (brect[2], brect[1] - 22),
                 (0, 0, 0), -1)

    if hand_sign_text != "":
        handedness_side = handedness_side + ':' + hand_sign_text
    cv.putText(image, handedness_side, (brect[0] + 5, brect[1] - 4),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv.LINE_AA)
    return image


def draw_info(image, fps, mode, number):
    cv.putText(image, f'FPS: {int(fps)}', (10, 30), cv.FONT_HERSHEY_COMPLEX,
               1, (255, 0, 0), 3)

    mode_string = ['Logging Key Point', 'Logging Point History']
    if 1 <= mode <= 2:
        cv.putText(image, "MODE:" + mode_string[mode - 1], (10, 90),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                   cv.LINE_AA)
        if 0 <= number <= 9:
            cv.putText(image, "NUM:" + str(number), (10, 110),
                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1,
                       cv.LINE_AA)
    return image


if __name__ == '__main__':
    main()
