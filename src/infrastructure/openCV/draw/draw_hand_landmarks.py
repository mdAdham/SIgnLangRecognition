import cv2 as cv
from numpy import ndarray


def draw_landmarks(image: ndarray, hand:  ndarray):
    if len(hand) > 0:
        draw_thumb(image, hand)
        draw_finger(image, hand, 5,6,7,8)
        draw_finger(image, hand, 9,10,11,12)
        draw_finger(image, hand, 13,14,15,16)
        draw_finger(image, hand, 17,18,19,20)
        draw_palm(image, hand)
        draw_hand_keypoints(image, hand)

    return image


def draw_hand_keypoints(image, knuckles: ndarray):
    for index, knuckle in enumerate(knuckles):
        if index == 0:  # 小指：指先
            cv.circle(image, (knuckle[0], knuckle[1]), 8, (255, 255, 255),
                      -1)
            cv.circle(image, (knuckle[0], knuckle[1]), 8, (0, 0, 0), 1)

        else:
            cv.circle(image, (knuckle[0], knuckle[1]), 5, (255, 255, 255),
                      -1)
            cv.circle(image, (knuckle[0], knuckle[1]), 5, (0, 0, 0), 1)


def draw_palm(image: ndarray, knuckles: ndarray):
    cv.line(image, tuple(knuckles[0]), tuple(knuckles[1]),
            (0, 0, 0), 6)
    cv.line(image,tuple(knuckles[0]), tuple(knuckles[1]),
            (255, 255, 255), 2)
    cv.line(image, tuple(knuckles[1]), tuple(knuckles[2]),
            (0, 0, 0), 6)
    cv.line(image,tuple(knuckles[1]), tuple(knuckles[2]),
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
    cv.line(image,  tuple(knuckles[17]), tuple(knuckles[0]),
            (255, 255, 255), 2)


def draw_finger(image: ndarray, knuckles: ndarray, wrist_joint: int, base_joint: int, mid_joint: int, tip_joint: int):
    cv.line(image, tuple(knuckles[wrist_joint]), tuple(knuckles[base_joint]),
            (0, 0, 0), 6)
    cv.line(image,tuple(knuckles[wrist_joint]), tuple(knuckles[base_joint]),
            (255, 255, 255), 2)
    cv.line(image, tuple(knuckles[base_joint]), tuple(knuckles[mid_joint]),
            (0, 0, 0), 6)
    cv.line(image, tuple(knuckles[base_joint]), tuple(knuckles[mid_joint]),
            (255, 255, 255), 2)
    cv.line(image, tuple(knuckles[mid_joint]), tuple(knuckles[tip_joint]),
            (0, 0, 0), 6)
    cv.line(image,  tuple(knuckles[mid_joint]), tuple(knuckles[tip_joint]),
            (255, 255, 255), 2)


def draw_thumb(image: ndarray, knuckles: ndarray):
    cv.line(image, tuple(knuckles[2]), tuple(knuckles[3]),
            (0, 0, 0), 6)
    cv.line(image,  tuple(knuckles[2]), tuple(knuckles[3]),
            (255, 255, 255), 2)
    cv.line(image,  tuple(knuckles[3]), tuple(knuckles[4]),
            (0, 0, 0), 6)
    cv.line(image, tuple(knuckles[3]), tuple(knuckles[4]),
            (255, 255, 255), 2)
