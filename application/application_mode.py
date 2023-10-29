from enum import Enum


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = ApplicationMode.PLAY
    if key == 107:  # k
        mode = ApplicationMode.LEARN_KEY_POINTS
    if key == 104:  # h
        mode = ApplicationMode.LEARN_POINT_HISTORY
    return number, mode


class ApplicationMode(Enum):
    PLAY = 1
    LEARN_KEY_POINTS = 2
    LEARN_POINT_HISTORY = 3
