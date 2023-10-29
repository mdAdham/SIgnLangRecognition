import mediapipe as mp


def initialize_mediapipe_hands(arguments) -> mp.solutions.hands.Hands:
    return mp.solutions.hands.Hands(
        static_image_mode=arguments.use_static_image_mode,
        max_num_hands=2,
        min_detection_confidence=arguments.min_detection_confidence,
        min_tracking_confidence=arguments.min_tracking_confidence,
    )
