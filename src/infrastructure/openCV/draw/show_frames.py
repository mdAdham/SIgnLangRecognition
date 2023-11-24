import cv2 as cv

def show_frame(debug_image_with_landmark_overlays):
    cv.imshow('Hand Gesture Recognition', debug_image_with_landmark_overlays.image)
