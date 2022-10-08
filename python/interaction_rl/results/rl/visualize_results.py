import os
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pickle

file_name = "training_record.npy"
training_record = np.load(file_name, allow_pickle=True).item()

# results data
success_record = training_record['success']
collision_record = training_record['collision']
# deflection_record = training_record['deflection']
time_exceed_record = training_record['time_exceed']
reward_record = training_record['episode_reward']

episode_num = 5000 # len(success_record)

# average data
avg_success_rate = []
avg_collision_rate = []
avg_deflection_rate = []
avg_time_exceed_rate = []
avg_episode_reward = []

average_range = 100
for i in range(1, episode_num):
    if i >= average_range:
        avg_success_rate.append(np.mean(success_record[i-average_range:i]))
        avg_collision_rate.append(np.mean(collision_record[i-average_range:i]))
        # avg_deflection_rate.append(np.mean(deflection_record[i-average_range:i]))
        avg_time_exceed_rate.append(np.mean(time_exceed_record[i-average_range:i]))
        avg_episode_reward.append(np.mean(reward_record[i-average_range:i]))
    else:
        avg_success_rate.append(np.mean(success_record[:i]))
        avg_collision_rate.append(np.mean(collision_record[:i]))
        # avg_deflection_rate.append(np.mean(deflection_record[:i]))
        avg_time_exceed_rate.append(np.mean(time_exceed_record[:i]))
        avg_episode_reward.append(np.mean(reward_record[:i]))

avg_result_record = {'success_rate': avg_success_rate,
                    'collision_rate': avg_collision_rate,
                    # 'deflection_rate': avg_deflection_rate,
                    'timeout_rate': avg_time_exceed_rate,
                    }

avg_reward_record = {'episode_reward': avg_episode_reward,
                    }

# fig, axes = plt.subplots()
# print(ax)
fig, axes = plt.subplots(2, 1)
result_figure = axes[0]
reward_figure = axes[1]
# plot result
for k,v in avg_result_record.items():
    result_figure.plot(range(1, episode_num), v, label=k)
result_figure.set_xlabel('Episodes')
result_figure.set_ylabel('Smoothed Episode Results(%)')
result_figure.set_title('TD3 Training Results Curve')
result_figure.legend()
# plot reward
for k,v in avg_reward_record.items():
    reward_figure.plot(range(1, episode_num), v, label=k)
reward_figure.set_xlabel('Episodes')
reward_figure.set_ylabel('Smoothed Episode Rewards')
reward_figure.set_title('TD3 Training Rewards Curve')
reward_figure.legend()

result_figure.grid(linestyle='-.')
reward_figure.grid(linestyle='-.')


result_figure.grid(linestyle='-.')
reward_figure.grid(linestyle='-.')
plt.show()
