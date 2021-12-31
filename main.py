#Imports
import numpy as np
import torch
import gym
import time
import pybullet_envs

from DDPG import DDPG
from utils import OrnsteinUhlenbeckProcess, beta_anneal

from replay_buffer import ReplayBuffer
from PER import PrioritizedReplayBuffer
import argparse

# 传进控制参数
parser = argparse.ArgumentParser()
# parser.add_argument('--seed', dest="seed", type=int)
parser.add_argument('--per', dest='prioritized', action='store_true')
# parser.add_argument('--env', dest='env')
parser.add_argument('--bn', dest="bn", action='store_true')
parser.add_argument('--notarget', dest="notarget", action='store_true')
parser.add_argument('--duel', dest="dueling", action='store_true')
args = parser.parse_args()
# prioritized = args.prioritized
# seed = args.seed
BN = args.bn
target = not args.notarget
dueling = args.dueling

prioritized = args.prioritized
# env_name = "InvertedPendulumBulletEnv-v0"
env_name = "HopperBulletEnv-v0"
# env_name = "CartPoleContinuousBulletEnv-v0"
# env_name = "Walker2DBulletEnv-v0"
# env_name = "ReacherBulletEnv-v0"
# env_name = "HalfCheetahBulletEnv-v0"
env = gym.make(env_name)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

# 设定种子，便于复现实验结果
seed = 0
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)

filename = env_name[:-12]
if not target:
    filename = filename + '_notarget'
if BN:
    filename = filename + '_BN'
if prioritized:
    filename += '_PER'
if dueling:
    filename += '_duel'
print('Running: '+ filename)

# 设定经验池
buffer_size = 1e6
if prioritized:
    replay_buffer = PrioritizedReplayBuffer(buffer_size) 
else:
    replay_buffer = ReplayBuffer(buffer_size)

net = DDPG(state_dim, action_dim, BN=BN, target=target, dueling=dueling)
test_rewards = [net.test(env)]
np.save("data2/{}".format(filename), test_rewards)


total_steps = 1000000
test_freq = 2000
test_steps = 0
start_time = time.time()
episode = 0
done = True
noise_type = "normal"
# noise_type = "O"
epsilon = 1e-6
beta = 0
beta_init = 0
beta_anneal_steps = total_steps

for train_steps in range(total_steps):
    # 如果游戏结束了
    if done:
        done = False
        if train_steps != 0:
            if train_steps > 3000:
                print("Steps: {} Reward: {} Time:{}".format(train_steps, episode_r, int(time.time() - start_time)))

            net.train(replay_buffer, prioritized, steps_to_done)

        if test_steps >= test_freq:
            test_steps %= test_freq
            test_reward = net.test(env)
            test_rewards.append(test_reward)
            np.save("data2/{}".format(filename), test_rewards)

        state = env.reset()
        episode_r = 0
        steps_to_done = 0
        episode += 1

    action = net.get_action(state)
    # 为当前的action加上噪声
    if noise_type == "normal":
        noise = np.random.normal(0, 0.1, size=action_dim)
    else:
        OUP = OrnsteinUhlenbeckProcess(size=action_dim, theta=0.15, sigma=2, mu=0);
        noise = OUP.sample()

    action = (action + noise).clip(env.action_space.low, env.action_space.high)
    next_state, reward, done, info = env.step(action)
    
    # 将新的经验添加到经验池中
    if not prioritized:
        replay_buffer.push((state, action, reward, next_state, done))
    else:
        replay_buffer.add(state, action, reward, next_state, done)

    # 更新
    state = next_state
    episode_r += reward
    test_steps += 1
    steps_to_done += 1
       

test_rewards.append(net.test(env))
np.save("data2/{}".format(filename), test_rewards)
print(test_rewards)
