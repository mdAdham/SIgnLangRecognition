from dataclasses import dataclass
import cv2 as cv


@dataclass
class Image:
    image: cv.typing.MatLike

    def lock(self):
        self.image.flags.writeable = False
        return self

    def unlock(self):
        self.image.flags.writeable = True
        return self

    def width(self) -> int:
        return self.image.shape[1]

    def height(self) -> int:
        return self.image.shape[0]
