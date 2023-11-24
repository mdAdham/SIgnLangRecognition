import cv2 as cv

from src.application.opencv.Image import Image


class VideoCapture:
    def __init__(self, arguments):
        self.capture = cv.VideoCapture(arguments.device)
        self.capture.set(cv.CAP_PROP_FRAME_WIDTH, arguments.width)
        self.capture.set(cv.CAP_PROP_FRAME_HEIGHT, arguments.height)

    def get_frame(self) -> tuple[bool, Image]:
        ret, image = self.capture.read()
        return ret, Image(image)

    def print_frame(self, debug_image_with_landmark_overlays):
        cv.imshow('Hand Gesture Recognition', debug_image_with_landmark_overlays.image)

    def destroy_windows(self):
        cv.destroyAllWindows()
