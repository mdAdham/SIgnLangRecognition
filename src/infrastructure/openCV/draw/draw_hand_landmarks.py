import cv2 as cv
from numpy import ndarray

from src.domain.Hands import Hand, Knuckle


def draw_landmarks(image: ndarray, hand: Hand):
    if len(hand.knuckles) > 0:
        draw_thumb(image, hand.knuckles)
        draw_finger(image, hand.knuckles, 5,6,7,8)
        draw_finger(image, hand.knuckles, 9,10,11,12)
        draw_finger(image, hand.knuckles, 13,14,15,16)
        draw_finger(image, hand.knuckles, 17,18,19,20)
        draw_palm(image, hand.knuckles)
        draw_hand_keypoints(image, hand.knuckles)

    return image


def draw_hand_keypoints(image, knuckles: list[Knuckle]):
    for index, knuckle in enumerate(knuckles):
        if index == 0:  # 小指：指先
            cv.circle(image, (knuckle.x, knuckle.y), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (knuckle.x, knuckle.y), 8, (0, 0, 0), 1)

        else:
            cv.circle(image, (knuckle.x, knuckle.y), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (knuckle.x, knuckle.y), 5, (0, 0, 0), 1)


def draw_palm(image: ndarray, knuckles: list[Knuckle]):
    cv.line(image, [knuckles[0].x, knuckles[0].y], [knuckles[1].x, knuckles[1].y],
            (0, 0, 0), 6)
    cv.line(image,[knuckles[0].x, knuckles[0].y], [knuckles[1].x, knuckles[1].y],
            (255, 255, 255), 2)
    cv.line(image, [knuckles[1].x, knuckles[1].y], [knuckles[2].x, knuckles[2].y],
            (0, 0, 0), 6)
    cv.line(image,[knuckles[1].x, knuckles[1].y], [knuckles[2].x, knuckles[2].y],
            (255, 255, 255), 2)
    cv.line(image, [knuckles[2].x, knuckles[2].y], [knuckles[5].x, knuckles[5].y],
            (0, 0, 0), 6)
    cv.line(image, [knuckles[2].x, knuckles[2].y], [knuckles[5].x, knuckles[5].y],
            (255, 255, 255), 2)
    cv.line(image, [knuckles[5].x, knuckles[5].y], [knuckles[9].x, knuckles[9].y],
            (0, 0, 0), 6)
    cv.line(image, [knuckles[5].x, knuckles[5].y], [knuckles[9].x, knuckles[9].y],
            (255, 255, 255), 2)
    cv.line(image, [knuckles[9].x, knuckles[9].y], [knuckles[13].x, knuckles[13].y],
            (0, 0, 0), 6)
    cv.line(image, [knuckles[9].x, knuckles[9].y], [knuckles[13].x, knuckles[13].y],
            (255, 255, 255), 2)
    cv.line(image, [knuckles[13].x, knuckles[13].y], [knuckles[17].x, knuckles[17].y],
            (0, 0, 0), 6)
    cv.line(image,[knuckles[13].x, knuckles[13].y], [knuckles[17].x, knuckles[17].y],
            (255, 255, 255), 2)
    cv.line(image, [knuckles[17].x, knuckles[17].y], [knuckles[0].x, knuckles[0].y],
            (0, 0, 0), 6)
    cv.line(image, [knuckles[17].x, knuckles[17].y], [knuckles[0].x, knuckles[0].y],
            (255, 255, 255), 2)


def draw_little_finger(image: ndarray, knuckles: list[Knuckle]):
    cv.line(image, [knuckles[17].x, knuckles[17].y], [knuckles[18].x, knuckles[18].y],
            (0, 0, 0), 6)
    cv.line(image, [knuckles[17].x, knuckles[17].y], [knuckles[18].x, knuckles[18].y],
            (255, 255, 255), 2)
    cv.line(image, [knuckles[10].x, knuckles[10].y], [knuckles[11].x, knuckles[11].y],
            (0, 0, 0), 6)
    cv.line(image, [knuckles[10].x, knuckles[10].y], [knuckles[11].x, knuckles[11].y],
            (255, 255, 255), 2)
    cv.line(image, [knuckles[11].x, knuckles[11].y], [knuckles[12].x, knuckles[12].y],
            (0, 0, 0), 6)
    cv.line(image, [knuckles[11].x, knuckles[11].y], [knuckles[12].x, knuckles[12].y],
            (255, 255, 255), 2)


def draw_ring_finger(image: ndarray, knuckles: list[Knuckle]):
    cv.line(image, [knuckles[13].x, knuckles[13].y], [knuckles[14].x, knuckles[14].y],
            (0, 0, 0), 6)
    cv.line(image, [knuckles[13].x, knuckles[13].y], [knuckles[14].x, knuckles[14].y],
            (255, 255, 255), 2)
    cv.line(image, [knuckles[14].x, knuckles[14].y], [knuckles[15].x, knuckles[15].y],
            (0, 0, 0), 6)
    cv.line(image, [knuckles[14].x, knuckles[14].y], [knuckles[15].x, knuckles[15].y],
            (255, 255, 255), 2)
    cv.line(image, [knuckles[15].x, knuckles[15].y], [knuckles[16].x, knuckles[16].y],
            (0, 0, 0), 6)
    cv.line(image, [knuckles[15].x, knuckles[15].y], [knuckles[16].x, knuckles[16].y],
            (255, 255, 255), 2)


def draw_middle_finger(image: ndarray, knuckles: list[Knuckle]):
    cv.line(image, [knuckles[9].x, knuckles[9].y], [knuckles[10].x, knuckles[10].y],
            (0, 0, 0), 6)
    cv.line(image, [knuckles[9].x, knuckles[9].y], [knuckles[10].x, knuckles[10].y],
            (255, 255, 255), 2)
    cv.line(image, [knuckles[10].x, knuckles[10].y], [knuckles[11].x, knuckles[11].y],
            (0, 0, 0), 6)
    cv.line(image, [knuckles[10].x, knuckles[10].y], [knuckles[11].x, knuckles[11].y],
            (255, 255, 255), 2)
    cv.line(image, [knuckles[11].x, knuckles[11].y], [knuckles[12].x, knuckles[12].y],
            (0, 0, 0), 6)
    cv.line(image, [knuckles[11].x, knuckles[11].y], [knuckles[12].x, knuckles[12].y],
            (255, 255, 255), 2)


def draw_finger(image: ndarray, knuckles: list[Knuckle], wrist_joint: int, base_joint: int, mid_joint: int, tip_joint: int):
    cv.line(image, [knuckles[wrist_joint].x, knuckles[wrist_joint].y], [knuckles[base_joint].x, knuckles[base_joint].y],
            (0, 0, 0), 6)
    cv.line(image,[knuckles[wrist_joint].x, knuckles[wrist_joint].y], [knuckles[base_joint].x, knuckles[base_joint].y],
            (255, 255, 255), 2)
    cv.line(image, [knuckles[base_joint].x, knuckles[base_joint].y], [knuckles[mid_joint].x, knuckles[mid_joint].y],
            (0, 0, 0), 6)
    cv.line(image, [knuckles[base_joint].x, knuckles[base_joint].y], [knuckles[mid_joint].x, knuckles[mid_joint].y],
            (255, 255, 255), 2)
    cv.line(image, [knuckles[mid_joint].x, knuckles[mid_joint].y], [knuckles[tip_joint].x, knuckles[tip_joint].y],
            (0, 0, 0), 6)
    cv.line(image, [knuckles[mid_joint].x, knuckles[mid_joint].y], [knuckles[tip_joint].x, knuckles[tip_joint].y],
            (255, 255, 255), 2)


def draw_thumb(image: ndarray, knuckles: list[Knuckle]):
    cv.line(image, [knuckles[2].x, knuckles[2].y], [knuckles[3].x, knuckles[3].y],
            (0, 0, 0), 6)
    cv.line(image, [knuckles[2].x, knuckles[2].y],[knuckles[3].x, knuckles[3].y],
            (255, 255, 255), 2)
    cv.line(image, [knuckles[3].x, knuckles[3].y], [knuckles[4].x, knuckles[4].y],
            (0, 0, 0), 6)
    cv.line(image, [knuckles[3].x, knuckles[3].y], [knuckles[4].x, knuckles[4].y],
            (255, 255, 255), 2)
