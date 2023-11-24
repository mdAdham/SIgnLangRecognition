from src.infrastructure.model.data_source.csv_client import log_key_points_to_csv, log_point_history_to_csv


def log_key_points(number, landmark_list):
    log_key_points_to_csv(number, landmark_list)


def log_point_history(number, point_history_list):
    log_point_history_to_csv(number, point_history_list)