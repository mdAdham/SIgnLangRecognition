import cv2 as cv

from application.opencv.Image import Image


def initialize_video_capture(arguments) -> cv.VideoCapture:
    capture = cv.VideoCapture(arguments.device)
    capture.set(cv.CAP_PROP_FRAME_WIDTH, arguments.width)
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, arguments.height)
    return capture


def read_image(capture) -> tuple[bool, Image]:
    ret, image = capture.read()
    return ret, Image(image)


def flip(image: Image) -> Image:
    return Image(cv.flip(image.image, 1))


def correct_color(flipped_image) -> Image:
    return Image(cv.cvtColor(flipped_image.image, cv.COLOR_BGR2RGB))


def show_frame(debug_image_with_landmark_overlays):
    cv.imshow('Hand Gesture Recognition', debug_image_with_landmark_overlays.image)


def destroy_windows():
    cv.destroyAllWindows()