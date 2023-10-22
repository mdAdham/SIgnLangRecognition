def draw_bounding_rectangle(use_bounding_rectangle, image, bounding_rectangle, cv):
    if use_bounding_rectangle:
        # Outer rectangle
        cv.rectangle(image, (bounding_rectangle[0], bounding_rectangle[1]),
                     (bounding_rectangle[2], bounding_rectangle[3]),
                     (0, 0, 0), 1)

    return image


def calculate_bounding_rectangle(image, landmarks, np, cv):
    image_width, image_height = image.shape[1], image.shape[0]

    landmark_array = np.empty((0, 2), int)

    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)

        landmark_point = [np.array((landmark_x, landmark_y))]

        landmark_array = np.append(landmark_array, landmark_point, axis=0)

    x, y, w, h = cv.boundingRect(landmark_array)

    return [x, y, x + w, y + h]
