import os
import tensorflow as tf
import numpy as np

from networks.rl_networks import AttActorNetwork, CriticNetwork
from config import hyperParameters

def get_split_batch(batch):
    '''memory.sample() returns a batch of experiences, but we want an array
    for each element in the memory (s, a, r, s', done)'''
    states_mb = np.array([each[0][0] for each in batch])
    # print(type(states_mb), states_mb.shape)
    actions_mb = np.array([each[0][1] for each in batch])
    if len(actions_mb.shape) < 2:
        actions_mb = actions_mb.reshape((*actions_mb.shape, 1))
    # print(actions_mb.shape)
    rewards_mb = np.array([each[0][2] for each in batch])
    # print(rewards_mb.shape)
    next_states_mb = np.array([each[0][3] for each in batch])
    # print(next_states_mb.shape)
    dones_mb = np.array([each[0][4] for each in batch])

    return states_mb, actions_mb, rewards_mb, next_states_mb, dones_mb

class ReinforcementAgent(object):
    def __init__(self, session, network_params, args):
        config = hyperParameters(args.control_steering)
        self.gamma = config.gamma
        self.td3_delay = config.td3_delay
        self.batch_size = config.batch_size

        print('Agent Network params:', network_params)
        self.model_name = 'AttTD3'
        self.actor = AttActorNetwork(name='actor', param_dict=network_params, control_steering=args.control_steering)
        self.critic_1 = CriticNetwork(name='critic_1', param_dict=network_params, control_steering=args.control_steering)
        self.critic_2 = CriticNetwork(name='critic_2', param_dict=network_params, control_steering=args.control_steering)
        print('Actor and Critics are all set')

        self.sess = session

        # record episode reward and result
        self.training_record = {}
        self.training_record['success'] = []
        self.training_record['collision'] = []
        self.training_record['time_exceed'] = []
        self.training_record['deflection'] = []
        self.training_record['episode_reward'] = []

        # network weights save folder
        self.saver = tf.compat.v1.train.Saver(max_to_keep=1000)
        self._root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._weights_dir = os.path.join(self._root_dir, 'models', 'rl', str(args.port))
        self._test_weights_dir = os.path.join(self._root_dir, 'models', 'rl', 'test')
        self._model_file_ending = '.ckpt'
        self._model_file = os.path.join(self._weights_dir, self.model_name + self._model_file_ending)

        # results record save folder
        self._record_dir = os.path.join(self._root_dir, 'results', 'rl', str(args.port))
        self._record_file = os.path.join(self._record_dir, 'training_record.npy')

    def initialize_model(self):
        self.sess.run(tf.compat.v1.global_variables_initializer())
        self.actor.update_target(self.sess)
        self.critic_1.update_target(self.sess)
        self.critic_2.update_target(self.sess)
        print('Network weights initialized')

    def load_model(self, episode=None, test=False):
        # load model weights
        if not episode: # load latest
            if test:
                model_file_with_ep = os.path.join(self._test_weights_dir, self.model_name + self._model_file_ending)
            else:
                model_file_with_ep = self._model_file
            self.saver = tf.compat.v1.train.import_meta_graph(model_file_with_ep + '.meta')
            self.saver.restore(self.sess, model_file_with_ep)
        else:
            if test:
                model_file_with_ep = os.path.join(self._test_weights_dir, self.model_name + '_' + str(episode) + self._model_file_ending)
            else:
                model_file_with_ep = os.path.join(self._weights_dir, self.model_name + '_' + str(episode) + self._model_file_ending)
            self.saver = tf.compat.v1.train.import_meta_graph(model_file_with_ep + '.meta')
            self.saver.restore(self.sess, model_file_with_ep)
        self.actor.update_target(self.sess)
        self.critic_1.update_target(self.sess)
        self.critic_2.update_target(self.sess)
        if self.saver is None:
            print("FAILED in loading network weights")
            return False
        print('Network weights loaded')
        # also load training record
        # self.training_record = np.load(self._record_file, allow_pickle=True)
        return True
    
    def save_model(self, episode=None):
        if not episode:
            self.saver.save(self.sess, self._model_file)
            print('Lateset model saved')
        else:
            model_file_with_ep = os.path.join(self._weights_dir, self.model_name + '_' + str(episode) + self._model_file_ending)
            self.saver.save(self.sess, model_file_with_ep)
            print(str(episode), ' episode model saved')


    def act(self, state, epsilon, is_testing=False):
        if is_testing:
            action = self.actor.get_action(self.sess, state=state, nosie=0)
        else:
            action = self.actor.get_action(self.sess, state=state, nosie=epsilon)

        return action

    def learn(self, buffer, epsilon):
        # "Delayed" Policy Updates
        for _ in range(self.td3_delay):
            # sample a batch data to update critic
            tree_idx, experimence, ISWeights_mb = buffer.sample(self.batch_size)
            # print('ISWeights_mb:', ISWeights_mb)
            # get q target values for next state from the target critic
            s_mb, a_mb, r_mb, next_s_mb, dones_mb = get_split_batch(experimence)
            a_target_next_state = self.actor.get_action_target(self.sess, next_s_mb, nosie=epsilon) # with Target Policy Smoothing
            q_target_next_state_1 = self.critic_1.get_q_value_target(self.sess, next_s_mb, a_target_next_state)
            q_target_next_state_2 = self.critic_2.get_q_value_target(self.sess, next_s_mb, a_target_next_state)
            q_target_next_state = np.minimum(q_target_next_state_1, q_target_next_state_2)
            # print('q_target_next_state:', q_target_next_state)

            # set Q_target = r if the episode ends at s+1, otherwise Q_target = r + gamma * Q_target(s',a')
            target_Qs_batch = []
            for i in range(0, len(dones_mb)):
                terminal = dones_mb[i]
                # if we are in a terminal state. only equals reward
                if terminal:
                    target_Qs_batch.append((r_mb[i]))
                else:
                    # take the Q taregt for action a'
                    target = r_mb[i] + self.gamma * q_target_next_state[i]
                    target_Qs_batch.append(target)
            targets_mb = np.array([each for each in target_Qs_batch])
            # print('target:', targets_mb)

            # critic train
            if len(a_mb.shape) > 2:
                a_mb = np.squeeze(a_mb, axis=1)
            loss, loss_2, absolute_errors = self.critic_1.train(self.sess, s_mb, a_mb, targets_mb, ISWeights_mb)
            _, _, absolute_errors_2 = self.critic_2.train(self.sess, s_mb, a_mb, targets_mb, ISWeights_mb)

            # update buffer priorities
            buffer.batch_update(tree_idx, absolute_errors)
        # actor train
        a_for_grad = self.actor.get_action_batch(self.sess, s_mb)
        a_gradients = self.critic_1.get_gradients(self.sess, s_mb, a_for_grad)
        self.actor.train(self.sess, s_mb, a_gradients[0])
        # target train
        self.actor.update_target(self.sess)
        self.critic_1.update_target(self.sess)
        self.critic_2.update_target(self.sess)
    
    def record_performence(self, episode_reward, aux_info):
        # record episode result and reward
        if aux_info['result'] == 'success':
            print('@ Success: reach the goal!!!')
            self.training_record['success'].append(1)
            self.training_record['collision'].append(0)
            self.training_record['time_exceed'].append(0)
            self.training_record['deflection'].append(0)
        elif aux_info['result'] == 'collision':
            print('@ Fail: collision')
            self.training_record['success'].append(0)
            self.training_record['collision'].append(1)
            self.training_record['time_exceed'].append(0)
            self.training_record['deflection'].append(0)
        elif aux_info['result'] == 'time_exceed':
            print('@ Fail: time exceed')
            self.training_record['success'].append(0)
            self.training_record['collision'].append(0)
            self.training_record['time_exceed'].append(1)
            self.training_record['deflection'].append(0)
        elif aux_info['result'] == 'deflection':
            print('@ Fail: deflection')
            self.training_record['success'].append(0)
            self.training_record['collision'].append(0)
            self.training_record['time_exceed'].append(0)
            self.training_record['deflection'].append(1)
        else:
            print('@ Unexpected end')
        self.training_record['episode_reward'].append(episode_reward)

        # save results record
        if not os.path.exists(self._record_dir):
            os.makedirs(self._record_dir)
        np.save(self._record_file, self.training_record)

        # print recent average results in terminal
        average_range = 100
        if len(self.training_record['episode_reward']) > average_range:
            success_rate_avg = np.mean(self.training_record['success'][-average_range:])
            collision_rate_avg = np.mean(self.training_record['collision'][-average_range:])
            time_exceed_rate_avg = np.mean(self.training_record['time_exceed'][-average_range:])
            deflection_rate_avg = np.mean(self.training_record['deflection'][-average_range:])
            reward_avg = np.mean(self.training_record['episode_reward'][-average_range:])
        else:
            success_rate_avg = np.mean(self.training_record['success'])
            collision_rate_avg = np.mean(self.training_record['collision'])
            time_exceed_rate_avg = np.mean(self.training_record['time_exceed'])
            deflection_rate_avg = np.mean(self.training_record['deflection'])
            reward_avg = np.mean(self.training_record['episode_reward'])
        print('[recent success {:.2f}, collision {:.2f}, time exceed {:.2f} and deflection {:.2f}]'.format(success_rate_avg, collision_rate_avg, time_exceed_rate_avg, deflection_rate_avg))
        print('[this episode reward is {:.2f}, recent episode reward is {:.2f}]'.format(episode_reward, reward_avg))
        print('___'*15)