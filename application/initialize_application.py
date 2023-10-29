from collections import deque

from application.application_mode import ApplicationMode
from infrastructure.argument_parser import get_arguments
from infrastructure.mediapipe.initialize_hands import initialize_mediapipe_hands
from infrastructure.openCV.video_capture.video_capture_lifecycle import initialize_video_capture
from model.keypoint_classifier.keypoint_classifier import KeyPointClassifier
from model.point_history_classifier.point_history_classifier import PointHistoryClassifier
from utils import CvFpsCalc


def initialize_application():
    arguments = get_arguments()

    history_length = 16

    return initialize_video_capture(arguments), CvFpsCalc(buffer_len=10), deque(
        maxlen=history_length), initialize_mediapipe_hands(
        arguments), KeyPointClassifier(), ApplicationMode.DEBUG, deque(maxlen=history_length), PointHistoryClassifier()
