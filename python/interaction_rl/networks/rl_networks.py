import numpy as np
import tensorflow as tf

def OU(action, mu=0, theta=0.15, sigma=0.3, dt=0.1):
    noise = theta * (mu - action) * dt + sigma * np.random.randn(1) * np.sqrt(dt)
    return noise

class AttActorNetwork():
    def __init__(self, name, param_dict, control_steering=True):
        # input and output size
        self.route_feature_num = param_dict['route_feature_num']
        self.ego_feature_num = param_dict['ego_feature_num']
        self.npc_feature_num = param_dict['npc_feature_num']
        self.npc_num = param_dict['npc_num']
        self.state_size = param_dict['state_size']
        if control_steering:
            self.action_size = param_dict['action_size']
        else:
            self.action_size = 1
        self.control_steering = control_steering
        
        # encoder and decoder size
        self.encoder_size = [64, 64]
        self.decoder_size = [256, 256]

        # attention head num and size
        self.features_per_head = 64
        self.feature_head = 1
        self.kqv_size = self.features_per_head * self.feature_head

        # learning rate and optimizer
        initial_learning_rate = param_dict['lra']
        global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                                        global_step=global_step,
                                                        decay_steps=50000,
                                                        decay_rate=0.75,
                                                        staircase=True)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)

        # build networks
        self.state_inputs, self.actor_variables, self.action, self.att_matrix = self.build_actor_network(name)
        self.state_inputs_target, self.actor_variables_target, self.action_target, self.att_matrix_target = self.build_actor_network(name + "_target")
        
        # update operations
        self.action_gradients = tf.compat.v1.placeholder(tf.float32, [None, self.action_size], name="action_gradients")
        self.actor_gradients = tf.compat.v1.gradients(self.action, self.actor_variables, -self.action_gradients)
        self.optimize = self.optimizer.apply_gradients(zip(self.actor_gradients, self.actor_variables), global_step=global_step)

        self.tau = param_dict['tau']
        self.update_target_op = [self.actor_variables_target[i].assign(tf.multiply(self.actor_variables[i], self.tau) + tf.multiply(self.actor_variables_target[i], 1 - self.tau)) 
                                for i in range(len(self.actor_variables))]

    def split_input(self, state):
        # state:[batch, ego_feature_num + npc_feature_num*npc_num + mask_num]
        mask = state[:, -(self.npc_num + 1):] # Dims: batch, len(mask)
        mask = mask < 0.5

        route_state = state[: , 0:self.route_feature_num]

        vehicle_state = state[:, self.route_feature_num:self.route_feature_num + self.ego_feature_num + self.npc_num * self.npc_feature_num] # Dims: batch, (ego+npcs)features
        ego_state = tf.reshape(vehicle_state[: , 0:self.ego_feature_num], [-1, 1, self.ego_feature_num]) # Dims: batch, 1, features
        npcs_state = tf.reshape(vehicle_state[: , self.ego_feature_num:], [-1, self.npc_num, self.npc_feature_num]) # Dims: batch, entities, features
        return ego_state, npcs_state, route_state, mask

    def attention(self, query, key, value, mask):
        """
            Compute a Scaled Dot Product Attention.
        :param query: size: batch, head, 1 (ego-entity), features
        :param key:  size: batch, head, entities, features
        :param value: size: batch, head, entities, features
        :param mask: size: batch,  head, 1 (absence feature), 1 (ego-entity)
        :return: the attention softmax(QK^T/sqrt(dk))V
        """
        scores = tf.matmul(query, tf.transpose(key, perm=[0, 1, 3, 2])) / np.sqrt(self.features_per_head)
        mask_constant = scores * 0 + -1e9
        if mask is not None:
            scores = tf.where(mask, mask_constant, scores)
        p_attn = tf.nn.softmax(scores, dim=-1)
        att_output = tf.matmul(p_attn, value)
        return att_output, p_attn

    def build_actor_network(self, name):
        with tf.compat.v1.variable_scope(name):
            state_inputs = tf.compat.v1.placeholder(tf.float32, [None, self.state_size], name="state_inputs")
            ego_state, npc_state, route_state, mask = self.split_input(state_inputs)

            # encode vehicle's state
            ego_encoder_1 = tf.layers.dense(inputs=ego_state,
                                            units=self.encoder_size[0],
                                            activation=tf.nn.tanh,
                                            kernel_initializer=tf.orthogonal_initializer(),
                                            name="ego_encoder_1")
            ego_encoder_2 = tf.layers.dense(inputs=ego_encoder_1,
                                            units=self.encoder_size[1],
                                            activation=tf.nn.tanh,
                                            kernel_initializer=tf.orthogonal_initializer(),
                                            name="ego_encoder_2") # Dims: batch, 1, size
            npc_encoder_1 = tf.layers.dense(inputs=npc_state,
                                            units=self.encoder_size[0],
                                            activation=tf.nn.tanh,
                                            kernel_initializer=tf.orthogonal_initializer(),
                                            name="npc_encoder_1")
            npc_encoder_2 = tf.layers.dense(inputs=npc_encoder_1,
                                            units=self.encoder_size[1],
                                            activation=tf.nn.tanh,
                                            kernel_initializer=tf.orthogonal_initializer(),
                                            name="npc_encoder_2") # Dims: batch, npc_num, size
            concat_encoder = tf.concat([ego_encoder_2, npc_encoder_2], axis=1) # Dims: batch, npc_num + 1, size

            # attention layer
            query_ego = tf.layers.dense(inputs=ego_encoder_2,
                                        units=self.kqv_size,
                                        use_bias=None,
                                        kernel_initializer=tf.orthogonal_initializer(),
                                        name="query_ego")
            key_all = tf.layers.dense(inputs=concat_encoder,
                                        units=self.kqv_size,
                                        use_bias=None,
                                        kernel_initializer=tf.orthogonal_initializer(),
                                        name="key_all")                                
            value_all = tf.layers.dense(inputs=concat_encoder,
                                        units=self.kqv_size,
                                        use_bias=None,
                                        kernel_initializer=tf.orthogonal_initializer(),
                                        name="value_all")
            # Dimensions: Batch, entity, head, feature_per_head
            query_ego = tf.reshape(query_ego, [-1, 1, self.feature_head, self.features_per_head])
            key_all = tf.reshape(key_all, [-1, self.npc_num + 1, self.feature_head, self.features_per_head])
            value_all = tf.reshape(value_all, [-1, self.npc_num + 1, self.feature_head, self.features_per_head])
            # Dimensions: Batch, head, entity, feature_per_head
            query_ego = tf.transpose(query_ego, perm=[0, 2, 1, 3])
            key_all = tf.transpose(key_all, perm=[0, 2, 1, 3])
            value_all = tf.transpose(value_all, perm=[0, 2, 1, 3])
            mask = tf.reshape(mask, [-1, 1, 1, self.npc_num + 1])
            mask = tf.tile(mask, [1, self.feature_head, 1, 1])
            # attention mechanism and its outcome
            att_result, att_matrix = self.attention(query_ego, key_all, value_all, mask)
            att_matrix = tf.identity(att_matrix, name="att_matrix")
            att_result = tf.reshape(att_result, [-1, self.features_per_head * self.feature_head], name = 'att_result')

            # encode route state
            if self.control_steering:
                route_encoder_1 = tf.layers.dense(inputs=route_state,
                                                units=self.encoder_size[0],
                                                activation=tf.nn.tanh,
                                                kernel_initializer=tf.orthogonal_initializer(),
                                                name="route_encoder_1")
                route_encoder_2 = tf.layers.dense(inputs=route_encoder_1,
                                                units=self.encoder_size[1],
                                                activation=tf.nn.tanh,
                                                kernel_initializer=tf.orthogonal_initializer(),
                                                name="route_encoder_2") # Dims: batch, size
            else:
                route_encoder_1 = tf.layers.dense(inputs=route_state,
                                                units=self.encoder_size[0]/2,
                                                activation=tf.nn.tanh,
                                                kernel_initializer=tf.orthogonal_initializer(),
                                                name="route_encoder_1")
                route_encoder_2 = tf.layers.dense(inputs=route_encoder_1,
                                                units=self.encoder_size[1]/2,
                                                activation=tf.nn.tanh,
                                                kernel_initializer=tf.orthogonal_initializer(),
                                                name="route_encoder_2") # Dims: batch, size
            concat_result = tf.concat([att_result, route_encoder_2], axis=1)      

            # decode attention and route result
            decoder_1 = tf.layers.dense(inputs=concat_result,
                                        units=self.decoder_size[0],
                                        activation=tf.nn.tanh,
                                        kernel_initializer=tf.variance_scaling_initializer(),
                                        name="decoder_1")
            decoder_2 = tf.layers.dense(inputs=decoder_1,
                                        units=self.decoder_size[1],
                                        activation=tf.nn.tanh,
                                        kernel_initializer=tf.variance_scaling_initializer(),
                                        name="decoder_2")
                                        
            # get actions
            if self.control_steering:
                acc = tf.layers.dense(inputs=decoder_2,
                                        units=1, activation=tf.nn.tanh,
                                        kernel_initializer=tf.variance_scaling_initializer(),
                                        name="acc")
                steer = tf.layers.dense(inputs=decoder_2,
                                        units=1, activation=tf.nn.tanh,
                                        kernel_initializer=tf.variance_scaling_initializer(),
                                        name="steer")
                action = tf.concat([acc, steer], axis=1, name="action")
            else:
                target_speed = tf.layers.dense(inputs=decoder_2,
                                                units=1, activation=tf.nn.sigmoid,
                                                kernel_initializer=tf.variance_scaling_initializer(),
                                                name="target_speed")
                action = tf.concat([target_speed], axis=1, name="action")
        actor_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        return state_inputs, actor_variables, tf.squeeze(action), att_matrix

    def get_attention_matrix(self, sess, state):
        if len(state.shape) < 2:
            state = state.reshape((1, *state.shape))
        attention_matrix = sess.run(self.attention_matrix, feed_dict={
                            self.state_inputs: state
                        })
        return attention_matrix

    def get_action(self, sess, state, nosie = 1):
        if nosie < 0:
            nosie = 0
        if len(state.shape) < 2:
            state = state.reshape((1, *state.shape))
        action = sess.run(self.action, feed_dict={
                            self.state_inputs: state
                        })
        if self.control_steering:
            acc_noised = action[0] + OU(action[0], mu=0, theta=0.3, sigma=0.45) * nosie
            steer_noised = action[1] + OU(action[1], mu=0, theta=0.15, sigma=0.3) * nosie
            action_noise = np.squeeze(np.array([np.clip(acc_noised, -1, 1), np.clip(steer_noised, -1, 1)]))
        else:
            target_speed_noised = action + OU(action, mu=0.5, theta=0.3, sigma=0.45) * nosie
            action_noise = np.clip(target_speed_noised, 0, 1)
        return action_noise

    def get_action_target(self, sess, state, nosie = 1):
        target_noise = 0.01
        if nosie < 0:
            nosie = 0
        action_target = sess.run(self.action_target, feed_dict={
                                    self.state_inputs_target: state
                                })

        if self.control_steering:
            action_target_smoothing = action_target + np.random.rand(self.action_size) * target_noise

            acc_smoothing = np.clip(action_target_smoothing[:, 0], -1, 1)
            acc_smoothing = acc_smoothing.reshape((*acc_smoothing.shape, 1))

            steer_smoothing = np.clip(action_target_smoothing[:, 1], -1, 1)
            steer_smoothing = steer_smoothing.reshape((*steer_smoothing.shape, 1))

            action_target_smoothing = np.concatenate([acc_smoothing, steer_smoothing, ], axis=1)
        else:
            action_target_smoothing = action_target + np.random.rand(self.action_size) * target_noise * nosie

            target_speed_smoothing = np.clip(action_target_smoothing[:], 0, 1)
            target_speed_smoothing = target_speed_smoothing.reshape((*target_speed_smoothing.shape, 1))

            action_target_smoothing = np.concatenate([target_speed_smoothing], axis=1)
        return action_target_smoothing

    def get_action_batch(self, sess, state):
        if len(state.shape) < 2:
            state = state.reshape((1, *state.shape))
        action = sess.run(self.action, feed_dict={
                            self.state_inputs: state
                        })
        return action

    def train(self, sess, state, action_gradients):
        sess.run(self.optimize, feed_dict={
            self.state_inputs: state,
            self.action_gradients: action_gradients
        })

    def update_target(self, sess):
        sess.run(self.update_target_op)

class CriticNetwork():
    def __init__(self, name, param_dict, control_steering):
        # input and output size
        self.route_feature_num = param_dict['route_feature_num']
        self.ego_feature_num = param_dict['ego_feature_num']
        self.npc_feature_num = param_dict['npc_feature_num']
        self.npc_num = param_dict['npc_num']
        self.state_size = param_dict['state_size']
        if control_steering:
            self.action_size = param_dict['action_size']
        else:
            self.action_size = 1
        self.control_steering = control_steering

        # encoder and decoder size
        self.encoder_size = [64, 64]
        self.decoder_size = [256, 256]
        
        # learning rate and optimizers
        initial_learning_rate = param_dict['lrc']
        global_step = tf.Variable(0, trainable=False)
        self.learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                                        global_step=global_step,
                                                        decay_steps=50000,
                                                        decay_rate=0.75,
                                                        staircase=True)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate)
        self.optimizer_2 = tf.compat.v1.train.AdamOptimizer(self.learning_rate)

        # build networks
        self.state_inputs, self.action, self.critic_variables, self.q_value = self.build_critic_network(name)
        self.state_inputs_target, self.action_target, self.critic_variables_target, self.q_value_target = self.build_critic_network(name + "_target")
        
        # update operations
        self.target = tf.compat.v1.placeholder(tf.float32, [None])
        self.ISWeights = tf.compat.v1.placeholder(tf.float32, [None, 1])
        self.absolute_errors = tf.abs(self.target - self.q_value)  # for updating sumtree
        self.action_gradients = tf.gradients(self.q_value, self.action)

        self.loss = tf.reduce_mean(self.ISWeights * tf.squared_difference(self.target, self.q_value))
        self.loss_2 = tf.reduce_mean(tf.compat.v1.losses.huber_loss(labels=self.target, predictions=self.q_value))
        self.optimize = self.optimizer.minimize(self.loss, global_step=global_step)
        self.optimize_2 = self.optimizer_2.minimize(self.loss_2,) #global_step=global_step)

        self.tau = param_dict['tau']
        self.update_target_op = [self.critic_variables_target[i].assign(tf.multiply(self.critic_variables[i], self.tau) + tf.multiply(self.critic_variables_target[i], 1 - self.tau)) for i in range(len(self.critic_variables))]

    def split_input(self, state):
        # state:[batch, ego_feature_num + npc_feature_num*npc_num + mask_num]
        mask = state[:, -(self.npc_num + 1):] # Dims: batch, len(mask)
        mask = mask < 0.5

        route_state = state[: , 0:self.route_feature_num]

        vehicle_state = state[:, self.route_feature_num:self.route_feature_num + self.ego_feature_num + self.npc_num * self.npc_feature_num] # Dims: batch, (ego+npcs)features
        return vehicle_state, route_state, mask

    def build_critic_network(self, name):
        with tf.compat.v1.variable_scope(name):
            state_inputs = tf.compat.v1.placeholder(tf.float32, [None, self.state_size], name="state_inputs")
            action_inputs = tf.compat.v1.placeholder(tf.float32, [None, self.action_size], name="action_inputs")
            vehicle_state, route_state, _ = self.split_input(state_inputs)
            # encode vehicles' state
            vehicle_encoder_1 = tf.layers.dense(inputs=vehicle_state,
                                            units=self.encoder_size[0],
                                            activation=tf.nn.tanh,
                                            kernel_initializer=tf.variance_scaling_initializer(),
                                            name="vehicle_encoder_1")
            vehicle_encoder_2 = tf.layers.dense(inputs=vehicle_encoder_1,
                                            units=self.encoder_size[1],
                                            activation=tf.nn.tanh,
                                            kernel_initializer=tf.variance_scaling_initializer(),
                                            name="vehicle_encoder_2")
            # encode route state
            route_encoder_1 = tf.layers.dense(inputs=route_state,
                                            units=self.encoder_size[0],
                                            activation=tf.nn.tanh,
                                            kernel_initializer=tf.orthogonal_initializer(),
                                            name="route_encoder_1")
            route_encoder_2 = tf.layers.dense(inputs=route_encoder_1,
                                            units=self.encoder_size[1],
                                            activation=tf.nn.tanh,
                                            kernel_initializer=tf.orthogonal_initializer(),
                                            name="route_encoder_2") # Dims: batch, size
            concat_state = tf.concat([vehicle_encoder_2, route_encoder_2], axis=1, name="concat_state")

            # encode ego's action
            concat_all = tf.concat([concat_state, action_inputs], axis=1, name="concat_all")

            # decode to q value
            decoder_1 = tf.layers.dense(inputs=concat_all,
                                        units=self.decoder_size[0], activation=tf.nn.tanh,
                                        kernel_initializer=tf.variance_scaling_initializer(),
                                        name="decoder_1")
            decoder_2 = tf.layers.dense(inputs=decoder_1,
                                        units=self.decoder_size[1], activation=tf.nn.tanh,
                                        kernel_initializer=tf.variance_scaling_initializer(),
                                        name="decoder_2")
            # q value output
            q_value = tf.layers.dense(inputs=decoder_2,
                                    units=1, activation=None,
                                    kernel_initializer=tf.variance_scaling_initializer(),
                                    name="q_value")
        critic_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, scope=name)
        return state_inputs, action_inputs, critic_variables, tf.squeeze(q_value)

    def get_q_value_target(self, sess, state, action):
        return sess.run(self.q_value_target, feed_dict={
            self.state_inputs_target: state,
            self.action_target: action
        })

    def get_gradients(self, sess, state, action):
        if len(action.shape) < 2:
            action = action.reshape((*action.shape, 1))
        return sess.run(self.action_gradients, feed_dict={
            self.state_inputs: state,
            self.action: action
        })

    def train(self, sess, state, action, target, ISWeights):
        _, _, loss, loss_2, absolute_errors = sess.run([self.optimize, self.optimize_2, self.loss, self.loss_2, self.absolute_errors], feed_dict={
            self.state_inputs: state,
            self.action: action,
            self.target: target,
            self.ISWeights: ISWeights
        })
        return loss, loss_2, absolute_errors

    def update_target(self, sess):
        sess.run(self.update_target_op)