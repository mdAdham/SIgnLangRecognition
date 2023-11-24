from dataclasses import dataclass
from enum import Enum


class Chirality(Enum):
    LEFT = "Left"
    RIGHT = "Right"


@dataclass
class Landmark:
    x: float
    y: float
    z: float


@dataclass
class Hand:
    landmarks: list[Landmark]
    handedness: Chirality


@dataclass
class Hands:
    hands_list: list[Hand]
