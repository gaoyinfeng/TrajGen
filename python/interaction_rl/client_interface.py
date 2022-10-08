import os
import zmq
import numpy as np
import random

from predict_trajectories.trajectory_loader import trajectory_loader

class ClientInterface(object):
    def __init__(self, args):
        self._context = zmq.Context()
        self.socket = self._context.socket(zmq.REQ)
        self.port = args.port
        print("connecting to interaction gym")
        url = ':'.join(["tcp://localhost", str(self.port)])
        self.socket.connect(url)

        self.args = args

        # simulator statue flags
        self.env_init_flag = False
        self.can_change_track_file_flag = False
        self.choose_ego_and_init_map_flag = False
        self.env_reset_flag = False
        
        # init predict trajectory(from stage 1) loader
        trajectory_files_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'predict_trajectories')
        if args.test:
            trajectory_files_path = os.path.join(trajectory_files_dir, self.args.scenario_name)
            self.trajectory_loader = trajectory_loader(trajectory_files_path, ego_num=1, test=True)
        else: # train
            trajectory_files_path = os.path.join(trajectory_files_dir, self.args.scenario_name)
            self.trajectory_loader = trajectory_loader(trajectory_files_path, ego_num=1)

        self.ego_id_list = None
        self.current_observation = None

    def __del__(self):
        self.socket.close()

    def env_init(self, args):
        setting = dict()
        setting['scenario_name'] = args.scenario_name
        setting['load_mode'] = args.load_mode

        setting['continous_action'] = args.continous_action
        setting['control_steering'] = args.control_steering
        setting['route_type'] = args.route_type

        setting['visualaztion'] = args.visualaztion
        setting['ghost_visualaztion'] = args.ghost_visualaztion
        setting['route_visualaztion'] = args.route_visualaztion
        setting['route_bound_visualaztion'] = args.route_bound_visualaztion

        setting['port'] = args.port
        
        # send to env
        message_send = {'command': 'env_init', 'content': setting}
        self.socket.send_string(str(message_send))
        
        # recieve from env
        message_recv = self.socket.recv_string()
        if message_recv == 'env_init_done':
            self.env_init_flag = True
            print('env init done')

    def change_track_file(self):
        track_dict = {}
        # send to env
        if self.args.route_type == 'predict':
            track_dict['track_type'] = 'predict'
            track_dict['track_content'] = self.trajectory_loader.get_trajectory_file_name()
            message_send = {'command': 'track_init', 'content': track_dict}
            self.socket.send_string(str(message_send))

        elif self.args.route_type == 'ground_truth' or self.args.route_type == 'centerline':
            track_dict['track_type'] = 'ground_truth'
            track_dict['track_content'] = self.track_file_index
            message_send = {'command': 'track_init', 'content': track_dict}
            self.socket.send_string(str(message_send))

        # recieve from env
        message_recv = self.socket.recv_string()
        if message_recv == 'change_file_done':
            self.can_change_track_file_flag = False

    def choose_ego_and_init_map(self):
        ego_info_dict = {}
        if self.args.route_type == 'predict':
            ego_info_dict['ego_id_list'] = self.trajectory_loader.get_ego_id()
            ego_info_dict['ego_start_timestamp'] = self.trajectory_loader.get_start_timestamp() # list
            ego_info_dict['ego_route'] = self.trajectory_loader.get_ego_routes() # dict
        elif self.args.route_type == 'ground_truth' or self.args.route_type == 'centerline':
            ego_info_dict['ego_id_list'] = self.trajectory_loader.get_ego_id()
            ego_info_dict['ego_start_timestamp'] = self.trajectory_loader.get_start_timestamp() # list
            ego_info_dict['ego_route'] = []

        # send to env
        message_send = {'command': 'ego_map_init', 'content': ego_info_dict}
        self.socket.send_string(str(message_send))

        # recieve from env
        message_recv = self.socket.recv()
        str_message = bytes.decode(message_recv)

        if str_message == 'wrong_num':
            print(message_recv)
        else:
            self.ego_id_list = eval(str_message)
            self.choose_ego_and_init_map_flag = True
    
    def reset_prepare(self):
        # init env
        if not self.env_init_flag:
            self.env_init(self.args)

        # change track file number
        elif self.env_init_flag and self.can_change_track_file_flag and not self.choose_ego_and_init_map_flag:
            if self.args.test:
                self.trajectory_loader.extract_data_from_file(random_select=False)
                self.track_file_index = self.trajectory_loader.get_csv_index()
            else: 
                self.trajectory_loader.extract_data_from_file(random_select=True)
                self.track_file_index = self.trajectory_loader.get_csv_index()
            self.change_track_file()

        # choose ego vehicle and map init
        elif self.env_init_flag and not self.choose_ego_and_init_map_flag:
            self.choose_ego_and_init_map()

        # several depandecies have been checked, can reset environment
        elif self.env_init_flag and self.choose_ego_and_init_map_flag and not self.env_reset_flag:
            return True

        return False

    def reset(self):
        # reset flags
        self.can_change_track_file_flag = True  # this is used for multi-track-file random selection
        self.choose_ego_and_init_map_flag = False
        self.env_reset_flag = False

        while not self.reset_prepare():
            self.reset_prepare()

        # send to env
        message_send = {'command': 'reset', 'content': self.env_reset_flag}
        self.socket.send_string(str(message_send))

        # recieve from env
        message_recv = self.socket.recv()
        message_recv = eval(bytes.decode(message_recv))
        
        if isinstance(message_recv, dict):
            self.env_reset_flag = True
            self.current_observation = message_recv['observation']
        else:
            self.choose_ego_and_init_map_flag = False
        state_dict_array = self.observation_to_ego_dict_array(self.current_observation, self.ego_id_list)
        return state_dict_array

    def step(self, action_dict):
        state_dict_array = self.observation_to_ego_dict_array(self.current_observation, self.ego_id_list)

        # send current action to env
        message_send = {'command': 'step', 'content': action_dict}
        self.socket.send_string(str(message_send))
        
        # recieve next observation, reward, done and aux info from env
        message_recv = self.socket.recv()
        message_recv = eval(bytes.decode(message_recv))

        self.current_observation = message_recv['observation']
        reward_dict = message_recv['reward']
        done_dict = message_recv['done']
        aux_info_dict = message_recv['aux_info']

        return state_dict_array, reward_dict, done_dict, aux_info_dict


    def observation_to_ego_dict_array(self, observation, ego_id_list):
        ego_state_dict = dict()
        for ego_id in ego_id_list:
            ego_state_dict[ego_id] = np.empty(shape=[0])

        if self.args.control_steering:
            state_order = ['lane_observation', 'target_speed', 'ego_next_pos', 'ego_shape', 'interaction_vehicles_observation', 'attention_mask']
        else:
            state_order = ['trajectory_pos', 'lane_observation', 'target_speed', 'current_speed', 'ego_next_pos', 'ego_shape', 'interaction_vehicles_observation', 'attention_mask']

        while state_order:
            for state_name, state_dict in observation.items():
                if state_name == state_order[0]:
                    for ego_id in ego_state_dict.keys():
                        ego_state_dict[ego_id] = np.append(ego_state_dict[ego_id], state_dict[ego_id])
                    state_order.pop(0)
                    break
        return ego_state_dict
