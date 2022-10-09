import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pickle

file_name = "testing_record.npy"
testing_record = np.load(file_name, allow_pickle=True).item()
success_files_name_list = []

total_test_num = 0
success_num = 0
time_exceed_num = 0
collision_num = 0

for track_file_name in testing_record.keys():
    track_record = testing_record[track_file_name]
    track_egos_list = list(track_record.keys())
    for ego_id in track_egos_list:
        ego_info = track_record[ego_id]
        if ego_info['collision'] is True:
            collision_num += 1
        elif ego_info['success'] is True:
            success_num += 1
            if track_file_name not in success_files_name_list:
                success_files_name_list.append(track_file_name)
        else:
            time_exceed_num += 1
            if track_file_name not in success_files_name_list:
                success_files_name_list.append(track_file_name)
        total_test_num += 1

print('total_test_num, success_num, time_exceed_num, collision_num:', total_test_num, success_num, time_exceed_num, collision_num)

