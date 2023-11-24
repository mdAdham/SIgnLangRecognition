from src.domain.Labels import KeyPointLabel


def read_gesture(finger_gesture_history, keypoint_classifier, landmark_list, point_history, point_history_classifier,
                 point_history_list, pre_processed_landmark_list) -> KeyPointLabel:

    hand_sign = keypoint_classifier(pre_processed_landmark_list)
    if hand_sign == KeyPointLabel.POINTER:
        point_history.append(landmark_list[8])
    else:
        point_history.append([0, 0])

    finger_gesture_id = 0
    point_history_len = len(point_history_list)
    if point_history_len == (point_history.maxlen.real * 2):
        finger_gesture_id = point_history_classifier(
            point_history_list)

    finger_gesture_history.append(finger_gesture_id)
    return hand_sign
