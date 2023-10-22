def initialize_video_capture(cv, arguments):
    capture = cv.VideoCapture(arguments.device)
    capture.set(cv.CAP_PROP_FRAME_WIDTH, arguments.width)
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, arguments.height)
    return capture