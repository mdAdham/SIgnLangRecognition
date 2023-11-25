import cv2 as cv
from numpy import ndarray

from src.application.opencv.Image import Image
from src.domain.Hands import Hand, Knuckle


def draw_landmarks(image: Image, hand: Hand):
    if len(hand.knuckles) > 0:
        draw_thumb(image.image, hand.knuckles)
        draw_index(image, hand.knuckles)
        draw_middle_finger(image, hand.knuckles)
        draw_ring_finger(image, hand.knuckles)
        draw_little_finger(image, hand.knuckles)
        draw_palm(image, hand.knuckles)
        draw_hand_keypoints(image, hand.knuckles)

    return image


def draw_hand_keypoints(image, knuckles: list[Knuckle]):
    for index, knuckle in enumerate(knuckles):
        if index == 0:  # 手首1
            cv.circle(image, (knuckle.x, knuckle.y), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (knuckle.x, knuckle.y), 5, (0, 0, 0), 1)
        if index == 1:  # 手首2
            cv.circle(image, (knuckle.x, knuckle.y), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (knuckle.x, knuckle.y), 5, (0, 0, 0), 1)
        if index == 2:  # 親指：付け根
            cv.circle(image, (knuckle.x, knuckle.y), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (knuckle.x, knuckle.y), 5, (0, 0, 0), 1)
        if index == 3:  # 親指：第1関節
            cv.circle(image, (knuckle.x, knuckle.y), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (knuckle.x, knuckle.y), 5, (0, 0, 0), 1)
        if index == 4:  # 親指：指先
            cv.circle(image, (knuckle.x, knuckle.y), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (knuckle.x, knuckle.y), 8, (0, 0, 0), 1)
        if index == 5:  # 人差指：付け根
            cv.circle(image, (knuckle.x, knuckle.y), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (knuckle.x, knuckle.y), 5, (0, 0, 0), 1)
        if index == 6:  # 人差指：第2関節
            cv.circle(image, (knuckle.x, knuckle.y), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (knuckle.x, knuckle.y), 5, (0, 0, 0), 1)
        if index == 7:  # 人差指：第1関節
            cv.circle(image, (knuckle.x, knuckle.y), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (knuckle.x, knuckle.y), 5, (0, 0, 0), 1)
        if index == 8:  # 人差指：指先
            cv.circle(image, (knuckle.x, knuckle.y), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (knuckle.x, knuckle.y), 8, (0, 0, 0), 1)
        if index == 9:  # 中指：付け根
            cv.circle(image, (knuckle.x, knuckle.y), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (knuckle.x, knuckle.y), 5, (0, 0, 0), 1)
        if index == 10:  # 中指：第2関節
            cv.circle(image, (knuckle.x, knuckle.y), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (knuckle.x, knuckle.y), 5, (0, 0, 0), 1)
        if index == 11:  # 中指：第1関節
            cv.circle(image, (knuckle.x, knuckle.y), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (knuckle.x, knuckle.y), 5, (0, 0, 0), 1)
        if index == 12:  # 中指：指先
            cv.circle(image, (knuckle.x, knuckle.y), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (knuckle.x, knuckle.y), 8, (0, 0, 0), 1)
        if index == 13:  # 薬指：付け根
            cv.circle(image, (knuckle.x, knuckle.y), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (knuckle.x, knuckle.y), 5, (0, 0, 0), 1)
        if index == 14:  # 薬指：第2関節
            cv.circle(image, (knuckle.x, knuckle.y), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (knuckle.x, knuckle.y), 5, (0, 0, 0), 1)
        if index == 15:  # 薬指：第1関節
            cv.circle(image, (knuckle.x, knuckle.y), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (knuckle.x, knuckle.y), 5, (0, 0, 0), 1)
        if index == 16:  # 薬指：指先
            cv.circle(image, (knuckle.x, knuckle.y), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (knuckle.x, knuckle.y), 8, (0, 0, 0), 1)
        if index == 17:  # 小指：付け根
            cv.circle(image, (knuckle.x, knuckle.y), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (knuckle.x, knuckle.y), 5, (0, 0, 0), 1)
        if index == 18:  # 小指：第2関節
            cv.circle(image, (knuckle.x, knuckle.y), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (knuckle.x, knuckle.y), 5, (0, 0, 0), 1)
        if index == 19:  # 小指：第1関節
            cv.circle(image, (knuckle.x, knuckle.y), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (knuckle.x, knuckle.y), 5, (0, 0, 0), 1)
        if index == 20:  # 小指：指先
            cv.circle(image, (knuckle.x, knuckle.y), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (knuckle.x, knuckle.y), 8, (0, 0, 0), 1)


def draw_palm(image: ndarray, knuckles: list[Knuckle]):
    cv.line(image, tuple(knuckles[0]), tuple(knuckles[1]),
            (0, 0, 0), 6)
    cv.line(image, tuple(knuckles[0]), tuple(knuckles[1]),
            (255, 255, 255), 2)
    cv.line(image, tuple(knuckles[1]), tuple(knuckles[2]),
            (0, 0, 0), 6)
    cv.line(image, tuple(knuckles[1]), tuple(knuckles[2]),
            (255, 255, 255), 2)
    cv.line(image, tuple(knuckles[2]), tuple(knuckles[5]),
            (0, 0, 0), 6)
    cv.line(image, tuple(knuckles[2]), tuple(knuckles[5]),
            (255, 255, 255), 2)
    cv.line(image, tuple(knuckles[5]), tuple(knuckles[9]),
            (0, 0, 0), 6)
    cv.line(image, tuple(knuckles[5]), tuple(knuckles[9]),
            (255, 255, 255), 2)
    cv.line(image, tuple(knuckles[9]), tuple(knuckles[13]),
            (0, 0, 0), 6)
    cv.line(image, tuple(knuckles[9]), tuple(knuckles[13]),
            (255, 255, 255), 2)
    cv.line(image, tuple(knuckles[13]), tuple(knuckles[17]),
            (0, 0, 0), 6)
    cv.line(image, tuple(knuckles[13]), tuple(knuckles[17]),
            (255, 255, 255), 2)
    cv.line(image, tuple(knuckles[17]), tuple(knuckles[0]),
            (0, 0, 0), 6)
    cv.line(image, tuple(knuckles[17]), tuple(knuckles[0]),
            (255, 255, 255), 2)


def draw_little_finger(image: ndarray, knuckles: list[Knuckle]):
    cv.line(image, tuple(knuckles[17]), tuple(knuckles[18]),
            (0, 0, 0), 6)
    cv.line(image, tuple(knuckles[17]), tuple(knuckles[18]),
            (255, 255, 255), 2)
    cv.line(image, tuple(knuckles[18]), tuple(knuckles[19]),
            (0, 0, 0), 6)
    cv.line(image, tuple(knuckles[18]), tuple(knuckles[19]),
            (255, 255, 255), 2)
    cv.line(image, tuple(knuckles[19]), tuple(knuckles[20]),
            (0, 0, 0), 6)
    cv.line(image, tuple(knuckles[19]), tuple(knuckles[20]),
            (255, 255, 255), 2)


def draw_ring_finger(image: ndarray, knuckles: list[Knuckle]):
    cv.line(image, tuple(knuckles[13]), tuple(knuckles[14]),
            (0, 0, 0), 6)
    cv.line(image, tuple(knuckles[13]), tuple(knuckles[14]),
            (255, 255, 255), 2)
    cv.line(image, tuple(knuckles[14]), tuple(knuckles[15]),
            (0, 0, 0), 6)
    cv.line(image, tuple(knuckles[14]), tuple(knuckles[15]),
            (255, 255, 255), 2)
    cv.line(image, tuple(knuckles[15]), tuple(knuckles[16]),
            (0, 0, 0), 6)
    cv.line(image, tuple(knuckles[15]), tuple(knuckles[16]),
            (255, 255, 255), 2)


def draw_middle_finger(image: ndarray, knuckles: list[Knuckle]):
    cv.line(image, tuple(knuckles[9]), tuple(knuckles[10]),
            (0, 0, 0), 6)
    cv.line(image, tuple(knuckles[9]), tuple(knuckles[10]),
            (255, 255, 255), 2)
    cv.line(image, tuple(knuckles[10]), tuple(knuckles[11]),
            (0, 0, 0), 6)
    cv.line(image, tuple(knuckles[10]), tuple(knuckles[11]),
            (255, 255, 255), 2)
    cv.line(image, tuple(knuckles[11]), tuple(knuckles[12]),
            (0, 0, 0), 6)
    cv.line(image, tuple(knuckles[11]), tuple(knuckles[12]),
            (255, 255, 255), 2)


def draw_index(image: ndarray, knuckles: list[Knuckle]):
    cv.line(image, tuple(knuckles[5]), tuple(knuckles[6]),
            (0, 0, 0), 6)
    cv.line(image, tuple(knuckles[5]), tuple(knuckles[6]),
            (255, 255, 255), 2)
    cv.line(image, tuple(knuckles[6]), tuple(knuckles[7]),
            (0, 0, 0), 6)
    cv.line(image, tuple(knuckles[6]), tuple(knuckles[7]),
            (255, 255, 255), 2)
    cv.line(image, tuple(knuckles[7]), tuple(knuckles[8]),
            (0, 0, 0), 6)
    cv.line(image, tuple(knuckles[7]), tuple(knuckles[8]),
            (255, 255, 255), 2)


def draw_thumb(image: ndarray, knuckles: list[Knuckle]):
    cv.line(image, tuple(knuckles[2]), tuple(knuckles[3]),
            (0, 0, 0), 6)
    cv.line(image, tuple(knuckles[2]), tuple(knuckles[3]),
            (255, 255, 255), 2)
    cv.line(image, tuple(knuckles[3]), tuple(knuckles[4]),
            (0, 0, 0), 6)
    cv.line(image, tuple(knuckles[3]), tuple(knuckles[4]),
            (255, 255, 255), 2)
