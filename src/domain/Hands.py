import itertools
from dataclasses import dataclass
from enum import Enum


class Chirality(Enum):
    LEFT = "Left"
    RIGHT = "Right"


@dataclass
class Knuckle:
    x: float
    y: float
    z: float


# https://developers.google.com/mediapipe/solutions/vision/hand_landmarker
@dataclass
class Hand:
    knuckles: list[Knuckle]
    handedness: Chirality

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


@dataclass
class Hands:
    hands_list: list[Hand]
