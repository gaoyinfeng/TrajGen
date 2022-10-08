import os
import time
import numpy as np
import argparse
import tensorflow as tf

from client_interface import ClientInterface
from config import hyperParameters
from agents.rl_agent import ReinforcementAgent
from replay_buffer.prioritized_replay import Buffer

def init_tensorflow():
    configProto = tf.compat.v1.ConfigProto()
    configProto.gpu_options.allow_growth = True
    # reset tensorflow graph
    tf.compat.v1.reset_default_graph()
    return configProto

def process_dict_to_experience(state_dict, action_dict, reward_dict, next_state_dict, done_dict):
    # usually there is only 1 car when training
    ego_id = list(state_dict.keys())[0]

    ex_state = state_dict[ego_id]
    ex_action = action_dict[ego_id]
    ex_reward = reward_dict[ego_id]
    ex_next_state = next_state_dict[ego_id]
    ex_done = done_dict[ego_id]

    return ex_state, ex_action, ex_reward, ex_next_state, ex_done

class main_loop(object):
    def __init__(self, args):
        self.args = args
        self.config = hyperParameters(control_steering=args.control_steering)
        self.interface = ClientInterface(args)

    def train_loop(self, continue_training=False):
        configProto = init_tensorflow()
        with tf.compat.v1.Session(config=configProto) as sess:
            print('Tensorflow session established')
            # set rl agent
            network_params = {'ego_feature_num': self.config.ego_feature_num,
                                'route_feature_num': self.config.route_feature_num,
                                'npc_feature_num': self.config.npc_feature_num,
                                'npc_num': self.config.npc_num,
                                'state_size': self.config.state_size,
                                'action_size': self.config.action_size,
                                'lra': self.config.lra,
                                'lrc': self.config.lrc,
                                'tau': self.config.tau,
                                } # a dict which contains a series of network params
            self.rl_agent = ReinforcementAgent(sess, network_params, self.args)

            # set replay buffer
            buffer = Buffer(self.config.buffer_size, self.config.pretrain_length)

            # initialize agent network and buffer
            if continue_training:
                self.rl_agent.load_model()
                buffer, current_total_steps, current_episode_num = buffer.load_buffer(continue_training=True)
            else:
                self.rl_agent.initialize_model()
                current_total_steps = 0
                current_episode_num = 0
                if self.config.load_buffer: 
                    buffer = buffer.load_buffer()
                else:
                    control_steering = self.args.control_steering
                    buffer.fill_buffer(self.interface, control_steering)
                    buffer.save_buffer(buffer)
            print('rl agent and replay buffer are initialized')

            # start interaction and training
            episode_num = current_episode_num
            total_steps = current_total_steps
            print('Starting training...')
            while not self.interface.socket.closed and episode_num <= self.config.max_episodes:
                # recording
                episode_reward = 0
                episode_step = 0
                episode_num += 1
                print('episode num:', episode_num)

                state_dict = self.interface.reset()
                while True:
                    epsilon = (self.config.noised_steps - total_steps)/self.config.noised_steps

                    action_dict = dict()
                    for ego_id, ego_state in state_dict.items():
                        action = self.rl_agent.act(ego_state, epsilon=epsilon)
                        action_dict[ego_id] = list(action)

                    next_state_dict, reward_dict, done_dict, aux_info_dict = self.interface.step(action_dict)

                    # store experimence, usually we use only 1 ego car for training
                    state_ex, action_ex, reward_ex, next_state_ex, done_ex = process_dict_to_experience(state_dict, action_dict, reward_dict, next_state_dict, done_dict)
                    experience = state_ex, action_ex, reward_ex, next_state_ex, done_ex
                    buffer.store(experience)

                    # episode reward ++
                    reward = list(reward_dict.values())[0]
                    episode_reward += reward

                    # update value and policy network, also update buffer priorities
                    if episode_step % self.config.learn_frequency == 0:
                        self.rl_agent.learn(buffer, epsilon=epsilon)
                    
                    # if a episode finished, print and record results; otherwise continue
                    if False not in done_dict.values(): # all egos are done
                        if epsilon > 0:
                            print('current epsilon is:', epsilon)
                        # save the model and replay buffer
                        if episode_num % self.config.model_save_frequency_latest == 0:
                            self.rl_agent.save_model()
                        if episode_num % self.config.model_save_frequency_regular == 0:
                            self.rl_agent.save_model(episode=episode_num)
                            buffer.save_buffer(buffer, episode_num=episode_num, total_steps=total_steps)
                        # record and print episode reward and result
                        aux_info = list(aux_info_dict.values())[0] # usually we use only 1 ego car to train
                        self.rl_agent.record_performence(episode_reward, aux_info)
                        break
                    else:
                        episode_step += 1
                        total_steps += 1
                        state_dict = next_state_dict


    def test_loop(self):
        configProto = init_tensorflow()
        with tf.compat.v1.Session(config=configProto) as sess:
            print('Tensorflow session established')
            # set rl agent
            network_params = {'ego_feature_num': self.config.ego_feature_num,
                                'route_feature_num': self.config.route_feature_num,
                                'npc_feature_num': self.config.npc_feature_num,
                                'npc_num': self.config.npc_num,
                                'state_size': self.config.state_size,
                                'action_size': self.config.action_size,
                                'lra': self.config.lra,
                                'lrc': self.config.lrc,
                                'tau': self.config.tau,
                                } # a dict which contains a series of network params
            self.rl_agent = ReinforcementAgent(sess, network_params, self.args)
            self.rl_agent.load_model(episode=None, test=True)
            print('rl agent is initialized')

            # set result recording dict and save path
            test_result_dict = dict()
            test_result_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'test')
            test_result_file = os.path.join(test_result_dir, 'testing_record.npy')

            # start interaction and testing
            print('Starting testing...')
            test_episode_num = np.inf
            episode_num = 0

            success = 0
            failure = 0
            collision = 0
            time_exceed = 0

            while not self.interface.socket.closed and episode_num <= test_episode_num:
                # recording
                episode_reward = 0
                episode_step = 0
                episode_num += 1
                
                # state reset
                state_dict = self.interface.reset()

                # set test result dict
                file_name = self.interface.trajectory_loader.file_name
                if file_name in test_result_dict.keys():
                    pass
                else:
                    test_result_dict[file_name] = dict()

                ego_id = self.interface.trajectory_loader.ego_id_list[0]
                test_result_dict[file_name][ego_id] = dict()
                ego_ground_truth_dict = self.interface.trajectory_loader.get_ego_ground_truth() # dict/list/list

                # start testing
                reach_list = []
                while True:
                    action_dict = dict()
                    for ego_id, ego_state in state_dict.items():
                        start_time = time.time()
                        action = self.rl_agent.act(ego_state, epsilon=-1)
                        action_dict[ego_id] = list(action)
                        if ego_id in reach_list:
                            action_dict[ego_id] = [-1.] # if the car reaches final point, hold still

                    next_state_dict, reward_dict, done_dict, aux_info_dict = self.interface.step(action_dict)

                    for ego_id, aux_info in aux_info_dict.items():
                        if aux_info['result'] == 'success':
                            reach_list.append(ego_id)

                    # episode reward ++, consider only 1 car for now
                    reward = list(reward_dict.values())[0]
                    episode_reward += reward
                    
                    # if a episode finish, print and record results; otherwise continue
                    if False not in done_dict.values(): # and list(aux_info_dict.values())[0] == 'time_exceed': # all egos are done
                        aux_info = list(aux_info_dict.values())[0]
                        # record results and trajectory
                        ego_trajectory = aux_info['trajectory']
                        if aux_info['result'] == 'collision':
                            test_result_dict[file_name][ego_id]['collision'] = True
                            test_result_dict[file_name][ego_id]['success'] = False
                            test_result_dict[file_name][ego_id]['trajectory'] = ego_trajectory
                            collision += 1
                            failure += 1
                        elif aux_info['result'] == 'time_exceed':
                            test_result_dict[file_name][ego_id]['collision'] = False
                            test_result_dict[file_name][ego_id]['success'] = False
                            test_result_dict[file_name][ego_id]['trajectory'] = ego_trajectory
                            time_exceed += 1
                        else:
                            test_result_dict[file_name][ego_id]['collision'] = False
                            test_result_dict[file_name][ego_id]['success'] = True
                            test_result_dict[file_name][ego_id]['trajectory'] = ego_trajectory
                            success += 1
                        # save records to disk
                        np.save(test_result_file, test_result_dict)

                        print(ego_id, "ego vehicle ended", 'Result:', aux_info['result'])
                        print('episode num:', episode_num, 'avoid collision:', success + time_exceed, 'collision:', collision)
                        print('----'*15)
                        break
                    else:
                        episode_step += 1
                        state_dict = next_state_dict

            print('-*'*15, ' result ', '-*'*15)
            print('success: ', success, '/', test_episode_num)
            print('collision: ', collision, '/', test_episode_num)
            print('time_exceed: ', time_exceed, '/', test_episode_num)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='TrajGen')
    parser.add_argument("scenario_name", type=str, default="DR_USA_Intersection_EP0", help="Name of the scenario (to identify map and folder for track files)", nargs='?')
    parser.add_argument("load_mode", type=str, default="vehicle", help="Dataset to load (vehicle, pedestrian, or both)", nargs='?')

    parser.add_argument("continous_action", type=bool, default=True, help="Is the action type continous or discrete", nargs='?')
    parser.add_argument("control_steering", type=bool, default=False, help="Control both lon and lat motions", nargs='?')
    parser.add_argument("route_type", type=str, default='predict', help="Default route type (predict, ground_truth or centerline)", nargs='?')

    parser.add_argument("visualaztion", type=bool, default=True, help="Visulize or not", nargs='?')
    parser.add_argument("ghost_visualaztion", type=bool, default=True, help="Render ghost ego or not", nargs='?')
    parser.add_argument("route_visualaztion", type=bool, default=True, help="Render ego's route or not", nargs='?')
    parser.add_argument("route_bound_visualaztion", type=bool, default=False, help="Render ego's route bound or not", nargs='?')
    
    parser.add_argument("--port", type=int, default=5557, help="Number of the port (int)")
    parser.add_argument('--continue_training', type=bool, default=False, help='wheather continue training')
    parser.add_argument('--test', action="store_true", default=False, help='testing single car')
    args = parser.parse_args()

    if args.scenario_name is None:
        raise IOError("You must specify a scenario. Type --help for help.")
    if args.load_mode != "vehicle" and args.load_mode != "pedestrian" and args.load_mode != "both":
        raise IOError("Invalid load command. Use 'vehicle', 'pedestrian', or 'both'")

    main_loop = main_loop(args)
    if args.test:
        main_loop.test_loop()
    else:
        main_loop.train_loop(args.continue_training)
