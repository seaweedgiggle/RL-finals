import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from replay_buffer import ReplayBuffer
from PER import PrioritizedReplayBuffer

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def fanin_init(size, fan_in=None):
    fan_in = fan_in or size[0]
    v = 1. / np.sqrt(fan_in)
    return torch.Tensor(size).uniform_(-v, v).cuda()

class Actor(nn.Module):
    def __init__(self, s_dim, a_dim, BN=False, fc1_out=400, fc2_out=300):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(s_dim, fc1_out)
        self.fc2 = nn.Linear(fc1_out, fc2_out)
        self.fc3 = nn.Linear(fc2_out, a_dim)
        self.BN = BN
        if self.BN:
            self.BN1 = nn.BatchNorm1d(s_dim)
            self.BN2 = nn.BatchNorm1d(fc1_out)
            self.BN3 = nn.BatchNorm1d(fc2_out)
            
        
    def init_weights(self, init_w=3e-3):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
  
    def forward(self, x):
        self.BN = self.BN and x.size()[0] > 1
#         if self.BN:
#             x = self.BN1(x)
        x = self.fc1(x)
        if self.BN:
            x = self.BN2(x)
        x = F.relu(x)
        x = self.fc2(x)
#         if self.BN:
#             x = self.BN3(x)
        x = F.relu(x)
        x = self.fc3(x)
        return torch.tanh(x)

class Critic(nn.Module):
    def __init__(self, s_dim, a_dim, BN=False, dueling=False, fc1_out=400, fc2_out=300):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(s_dim, fc1_out)
        self.fc2 = nn.Linear(fc1_out + a_dim, fc2_out)
        self.fc3 = nn.Linear(fc2_out, 1)
        self.BN = BN
        self.action_dim = a_dim
        self.dueling = dueling
        if self.BN:
            self.BN1 = nn.BatchNorm1d(s_dim)
            self.BN2 = nn.BatchNorm1d(fc1_out)
        if self.dueling:
            self.advantage = nn.Sequential(
                nn.Linear(fc1_out + a_dim, 300),
                nn.ReLU(),
                nn.Linear(300, fc1_out + a_dim)
            )
            
            self.value = nn.Sequential(
                nn.Linear(fc1_out + a_dim, 300),
                nn.ReLU(),
                nn.Linear(300, 1)
            )
    
    def init_weights(self, init_w=3e-3):
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3.weight.data.uniform_(-init_w, init_w)
        
    def forward(self, x, a):
        self.BN = self.BN and x.size()[0] > 1
#         if self.BN:
#             x = self.BN1(x)
        x = self.fc1(x)
#         if self.BN:
#             x = self.BN2(x)
        x = F.relu(x)
        x = torch.cat([x, a], 1)
        if self.dueling:
            x = self.advantage(x)
            x = self.value(x)
            return x
        else:  
            x = F.relu(self.fc2(x))
            x = self.fc3(x)
            return x 

class DDPG(object):
    def __init__(self, s_dim, a_dim, BN=False, target=True, dueling=False):
        # actor部分
        self.actor = Actor(s_dim, a_dim, BN).to(device)
        self.actor_target = Actor(s_dim, a_dim, BN).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        
        # critic部分
        self.critic = Critic(s_dim, a_dim, BN, dueling).to(device)
        self.critic_target = Critic(s_dim, a_dim, BN, dueling).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=1e-2)

        self.test_max_episode_length = None
        self.target = target
    
    # 输入一个状态state，通过actor输出一个动作action
    def get_action(self, s):
        s = torch.FloatTensor(np.array(s)).view(1, -1).to(device)
        action = self.actor(s).cpu().data.numpy().flatten()
        return action

    # 训练过程
    def train(self, replay_buffer, prioritized, steps, beta_value=0, epsilon=1e-6, batch_size=64, gamma=0.99, tau=0.005):
        self.actor.train()
        self.critic.train()
        if self.target == False:
            tau = 1
        for i in range(steps):
            if prioritized: 
                experience = replay_buffer.sample(batch_size, beta_value)
                s, a, r, s_new, done, weights, batch_idxes = experience
                r = r.reshape(-1, 1)
                done = done.reshape(-1, 1)
                weights = np.ones_like(r)
                weights = np.sqrt(weights)
            else:
                s, a, r, s_new, done = replay_buffer.sample(batch_size)
                weights, batch_idxes = np.ones_like(r), None
                weights = np.sqrt(weights)
            
            
            state = torch.FloatTensor(s).to(device)
            action = torch.FloatTensor(a).to(device)
            next_state = torch.FloatTensor(s_new).to(device)
            done = torch.FloatTensor(1 - done).to(device)
            reward = torch.FloatTensor(r).to(device)
            weights = torch.FloatTensor(weights).to(device)
   
            # 从actor与critic的目标网络中计算目标Q值
            Q_target = self.critic_target(next_state, self.actor_target(next_state))
            
            # 从critic网络估计Q值
            Q = self.critic(state, action)
            
            # 计算TD error
            Y = reward + (done * gamma * Q_target).detach()
            TD_errors = (Y - Q)
            
            # 计算带权重的TD error
            weighted_TD_errors = torch.mul(TD_errors, weights).to(device)
            
            zero_tensor = torch.zeros(weighted_TD_errors.shape).to(device)
            
            # 计算loss
            critic_loss = F.mse_loss(weighted_TD_errors, zero_tensor)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            actor_loss = -self.critic(state, self.actor(state)).mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # 更新目标网络
            for critic_weights, critic__target_weights in zip(self.critic.parameters(), self.critic_target.parameters()):
                critic__target_weights.data.copy_(tau * critic_weights.data + (1 - tau) * critic__target_weights.data)
            for actor_weights, actor__target_weights in zip(self.actor.parameters(), self.actor_target.parameters()):
                actor__target_weights.data.copy_(tau * actor_weights.data + (1 - tau) * actor__target_weights.data)
    
            #更新PER的优先值
            if prioritized:
                td_errors = TD_errors.cpu().detach().numpy()
                new_priorities = np.abs(td_errors) + epsilon
                replay_buffer.update_priorities(batch_idxes, new_priorities)

    # 测试过程
    def test(self, env, num_episodes=10):
        self.actor.eval()
        self.critic.eval()
        episode_rewards = []
        for episode in range(num_episodes):
            state = env.reset()
            steps = 0
            rewards = 0
            done = False
            while not done:
                # 根据当前状态，获取下一时刻动作
                action = self.get_action(state)

                next_state, reward, done, info = env.step(action)
                state = next_state
                rewards += reward
                steps += 1

                # 预防出现死循环，结束不了一个episode的情况
                if self.test_max_episode_length and steps >= self.test_max_episode_length -1:
                    done = True

            episode_rewards.append(rewards)
        avg_rewards = np.mean(np.array(episode_rewards))
        print("average reward in testing: {}".format(avg_rewards))
        return avg_rewards

        

