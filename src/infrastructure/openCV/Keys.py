import cv2 as cv


def get_key_press() -> int:
    return cv.waitKey(10)
