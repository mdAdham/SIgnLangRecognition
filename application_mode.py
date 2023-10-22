from enum import Enum

application_mode = Enum('application_mode', ['PLAY', 'LEARN_KEY_POINTS', 'LEARN_POINT_HISTORY'])


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = application_mode['PLAY']
    if key == 107:  # k
        mode = application_mode['LEARN_KEY_POINTS']
    if key == 104:  # h
        mode = application_mode['LEARN_POINT_HISTORY']
    return number, mode
