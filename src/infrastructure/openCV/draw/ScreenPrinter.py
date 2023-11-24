import numpy as np

from src.application.opencv.Image import Image


class ScreenPrinter:
    def __init__(self, cv, image: Image):
        self.cv = cv
        self.image = image

    def draw_rectangle(self, landmarks):
        bounding_rectangle = self.calculate_bounding_rectangle(landmarks)
        self.cv.rectangle(self.image, (bounding_rectangle[0], bounding_rectangle[1]),
                          (bounding_rectangle[2], bounding_rectangle[3]),
                          (0, 0, 0), 1)

    def calculate_bounding_rectangle(self, landmarks):
        landmark_array = np.empty((0, 2), int)

        for _, landmark in enumerate(landmarks.landmark):
            landmark_x = min(int(landmark.x * self.image.width()), self.image.width() - 1)
            landmark_y = min(int(landmark.y * self.image.height()), self.image.height() - 1)

            landmark_point = [np.array((landmark_x, landmark_y))]

            landmark_array = np.append(landmark_array, landmark_point, axis=0)

        x, y, w, h = self.cv.boundingRect(landmark_array)

        return [x, y, x + w, y + h]
