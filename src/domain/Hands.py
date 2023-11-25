import itertools
from collections import deque
from dataclasses import dataclass
from enum import Enum

from numpy import ndarray


class Chirality(Enum):
    LEFT = "Left"
    RIGHT = "Right"


@dataclass
class Knuckle:
    x: float
    y: float
    z: float = 0.0


# https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
class Hand:
    knuckles: list[Knuckle]
    handedness: Chirality
    history_length = 16
    point_history: deque = deque(maxlen=history_length)

    def __init__(self, knuckles: list[Knuckle], handedness: Chirality):
        self.knuckles = knuckles
        self.handedness = handedness

    def get_index(self) -> Knuckle:
        return self.knuckles[8]

    def get_base(self) -> Knuckle:
        return self.knuckles[0]

    def convert_to_relative_coordinates(self) -> list:
        relative_knuckle_list: list = []
        base_x, base_y = 0, 0
        for index, knuckle in enumerate(self.knuckles):
            if index == 0:
                base_x, base_y = knuckle.x, knuckle.y

            relative_knuckle_list.append([knuckle.x - base_x, knuckle.y - base_y])
        return relative_knuckle_list

    def flatten_list(self, knuckle_list) -> list:
        return list(itertools.chain.from_iterable(knuckle_list))

    def normalize(self, knuckle_list) -> list:
        max_value = max(list(map(abs, knuckle_list)))

        def normalize_(n):
            return n / max_value

        return list(map(normalize_, knuckle_list))

    def prepare_for_model(self) -> list:
        return self.normalize(self.flatten_list(self.convert_to_relative_coordinates()))

    def prepare_points_for_model(self) -> list:
        return self.flatten_list(self.convert_to_relative_coordinates())


@dataclass
class Hands:
    hands_list: list[Hand]
