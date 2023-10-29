import csv


def log_key_points_to_csv(number, landmark_list):
    with open('model/keypoint_classifier/keypoint.csv', 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([number, *landmark_list])


def log_point_history_to_csv(number, point_history_list):
    with open('model/point_history_classifier/point_history.csv', 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([number, *point_history_list])
