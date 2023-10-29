import mediapipe as mp

from application.opencv.Image import Image


def process_image(hands: mp.solutions.hands.Hands, image: Image):
    return hands.process(image.image)
