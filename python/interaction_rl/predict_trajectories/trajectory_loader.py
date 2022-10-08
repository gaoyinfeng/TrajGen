import os
import random
import pickle
import numpy as np

class trajectory_loader(object):
    def __init__(self, trajectory_files_path, ego_num, test=False):
        self.trajectory_files_dir_path = trajectory_files_path
        self.trajectory_files_dir_path_in_docker = '/home/developer/workspace/interaction-dataset-master/python/interaction_rl/predict_trajectories/' + trajectory_files_path.split('/')[-1]
        self.files = os.listdir(self.trajectory_files_dir_path)

        self.file_index = 0
        self.data = None

        self.ego_num = ego_num
        self.test = test

    def random_select_trajectroy_files(self, file_num=1):
        file_list = random.sample(self.files, file_num)
        self.file_name = random.choice(file_list)
        print(self.file_name)

        self.trajectory_file = os.path.join(self.trajectory_files_dir_path, self.file_name)
        with open(self.trajectory_file, 'rb') as f:
            self.data = pickle.load(f, encoding='latin1')
    
    def specific_select_trajectory_files(self):
        while True:
            # test all
            file_list = [self.files[self.file_index]]
            self.file_index += 1
            self.file_name = random.choice(file_list)
            print(self.file_name)

            self.trajectory_file = os.path.join(self.trajectory_files_dir_path, self.file_name)
            with open(self.trajectory_file, 'rb') as f:
                self.data = pickle.load(f, encoding='latin1')
                # print(self.data['others_track'].keys())
            
            if self.test:
                if self.ego_num == 1:
                    if self.data['gt_of_trouble']: # test data file still has collision/rotate/acc exceed events
                        break
                else:
                    break
        

    def extract_data_from_file(self, random_select=True):
        if random_select:
            self.random_select_trajectroy_files(file_num=1)
        else:
            if self.ego_num == 1:
                if self.data and self.data['gt_of_trouble']: # if current data exist and still have unrevised trouble cars
                    pass
                else:
                    self.specific_select_trajectory_files()
            else:
                self.specific_select_trajectory_files()
    
    def get_trajectory_file_name(self):
        self.trajectory_file_in_docker = os.path.join(self.trajectory_files_dir_path_in_docker, self.file_name)
        return self.trajectory_file_in_docker

    def get_csv_index(self):
        csv_index = self.trajectory_file[self.trajectory_file.find('test_1201') + 10]
        return csv_index

    def get_ego_id(self):
        self.ego_id_list = []
        if self.test:
            # if we want to test trouble trajectories, only the collision/rotate/acc exceed ones are selected
            if self.ego_num == 1:
                for ego_id in self.data['gt_of_trouble'].keys():
                    self.ego_id_list.append(ego_id)
                    break
            # else we selected multiple cars as ego
            else:
                for ego_id in self.data['egos_track'].keys():
                    self.ego_id_list.append(ego_id)
                    if len(self.ego_id_list) == self.ego_num:
                        break
        # otherwise conditions such as training, we random select vehicle within all vehicles
        else:
            self.ego_id_list.append(random.choice(list(self.data['egos_track'].keys())))

        return self.ego_id_list

    def get_start_timestamp(self):
        for info in self.data['egos_track'].values():
            ego_start_timestamp_list = [info[0][0]]
        return ego_start_timestamp_list

    def get_ego_routes(self):
        ego_route_dict = dict()

        for ego_id in self.ego_id_list:
            ego_info = self.data['egos_track'][ego_id]
            ego_route_dict[ego_id] = ego_info[1:]

        return ego_route_dict

    def get_ego_ground_truth(self):
        ego_ground_truth_dict = dict()

        for ego_id in self.ego_id_list:
            ego_info = self.data['gt_of_trouble'][ego_id]
            ego_ground_truth_dict[ego_id] = ego_info
            self.data['gt_of_trouble'].pop(ego_id)

        return ego_ground_truth_dict



