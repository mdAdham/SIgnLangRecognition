from infrastructure.openCV.draw.draw_debug_messages import draw_bounding_rectangle, draw_info_text, calculate_bounding_rectangle, \
    draw_point_history, draw_statistics
from infrastructure.openCV.draw.draw_hand_landmarks import draw_landmarks


def draw_overlays_with_landmarks(debug_image, hand_sign_id, handedness, keypoint_classifier_labels, landmark_list,
                                 most_common_fg_id, point_history_classifier_labels, use_bounding_rectangle,
                                 hand_landmarks, cv):
    bounding_rectangle = calculate_bounding_rectangle(debug_image, hand_landmarks, cv)
    debug_image = draw_bounding_rectangle(use_bounding_rectangle, debug_image, bounding_rectangle, cv)
    debug_image = draw_landmarks(debug_image, landmark_list, cv)
    debug_image = draw_info_text(
        debug_image,
        bounding_rectangle,
        handedness,
        keypoint_classifier_labels[hand_sign_id],
        point_history_classifier_labels[most_common_fg_id[0][0]],
        cv
    )
    return debug_image


def draw_overlays(debug_image, fps, mode, number, point_history, cv):
    debug_image = draw_point_history(debug_image, point_history, cv)
    debug_image = draw_statistics(debug_image, fps, mode, number, cv)
    return debug_image
