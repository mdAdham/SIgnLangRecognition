from src.application.application_mode import ApplicationMode
from src.infrastructure.argument_parser import get_arguments
from src.infrastructure.mediapipe.HandsReader import HandsReader
from src.infrastructure.model.GestureReader import GestureReader
from src.infrastructure.openCV.draw.ScreenPrinter import ScreenPrinter
from src.infrastructure.openCV.video_capture.VideoCaptor import VideoCapture
from src.utils import CvFpsCalc


def initialize_application() -> [VideoCapture, CvFpsCalc, HandsReader, GestureReader, ScreenPrinter, ApplicationMode]:
    arguments = get_arguments()
    use_static_image = False
    return VideoCapture(arguments), CvFpsCalc(buffer_len=10), \
        HandsReader(use_static_image,
                    arguments.min_detection_confidence,
                    arguments.min_tracking_confidence), \
        GestureReader(), \
        ScreenPrinter(), \
        ApplicationMode.DEBUG
