from dataclasses import dataclass
import cv2 as cv
import copy

from numpy import ndarray


@dataclass
class Image:
    image: ndarray

    def prepare(self):
        flipped_image = self.flip()  # Mirror display

        return flipped_image.correct_color(), copy.deepcopy(flipped_image)

    def flip(self):
        return Image(cv.flip(self.image, 1))

    def correct_color(self):
        return Image(cv.cvtColor(self.image, cv.COLOR_BGR2RGB))

    def width(self) -> int:
        return self.image.shape[1]

    def height(self) -> int:
        return self.image.shape[0]
