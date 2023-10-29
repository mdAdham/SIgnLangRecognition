from dataclasses import dataclass
import cv2 as cv


@dataclass
class Image:
    image: cv.typing.MatLike

    def width(self) -> int:
        return self.image.shape[1]

    def height(self) -> int:
        return self.image.shape[0]
