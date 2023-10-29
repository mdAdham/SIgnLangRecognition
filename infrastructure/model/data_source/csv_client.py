import csv


def read_from_csv():
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]
    return keypoint_classifier_labels, point_history_classifier_labels


def log_key_points_to_csv(number, landmark_list):
    with open('model/keypoint_classifier/keypoint.csv', 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([number, *landmark_list])


def log_point_history_to_csv(number, point_history_list):
    with open('model/point_history_classifier/point_history.csv', 'a', newline="") as f:
        writer = csv.writer(f)
        writer.writerow([number, *point_history_list])
