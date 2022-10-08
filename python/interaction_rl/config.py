
class hyperParameters(object):
    def __init__(self, control_steering):
        # about agent's input and output dimention
        if control_steering:
            self.action_size = 2
            self.route_feature_num = 9
            self.ego_feature_num = 4
        else:
            self.action_size = 1
            self.route_feature_num = 8
            self.ego_feature_num = 5
        
        self.npc_num = 5
        self.npc_feature_num = 7
        self.mask_num = self.npc_num + 1 # indicating vehicles' present
        self.state_size = self.route_feature_num + self.ego_feature_num + self.npc_num*self.npc_feature_num + self.mask_num

        self.max_episodes = 100000
        self.max_steps = 100
        self.noised_steps = 100000 # 100k step of explore
        self.gamma = 0.99  # Discounting rate
        self.learn_frequency = 2
        self.td3_delay = 2

        self.lra= 1e-5
        self.lrc= 5e-5
        self.batch_size = 256
        self.tau = 1e-3

        # replay buffer
        self.pretrain_length = 2000 # 2k random actions 
        self.buffer_size = 200000 # 100k
        self.load_buffer = False

        # model saving
        self.model_save_frequency_latest = 50
        self.model_save_frequency_regular = 250