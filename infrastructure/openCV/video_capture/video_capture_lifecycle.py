import cv2 as cv


def initialize_video_capture(arguments) -> cv.VideoCapture:
    capture = cv.VideoCapture(arguments.device)
    capture.set(cv.CAP_PROP_FRAME_WIDTH, arguments.width)
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, arguments.height)
    return capture
