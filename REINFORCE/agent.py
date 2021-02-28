import sys
sys.path.append("REINFORCE")
import network
import torch
from torch import nn
from torch import optim
import numpy as np
import os
from torch.distributions import Bernoulli
from torch.autograd import Variable
import random
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)
class PG_agent():
    def __init__(self, obs_dim, num_act, lr):
        self.obs_dim = obs_dim
        self.num_act = num_act 
        self.lr = lr
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = network.PGnetwork(self.obs_dim, self.num_act)
        self.optimizer = optim.Adam(self.model.parameters(), self.lr)
    def sample(self, obs):
        with torch.no_grad():
            obs = np.expand_dims(obs, axis = 0)
            obs = torch.tensor(obs, dtype = torch.float32).to(self.device)
            self.model.to(self.device)
            self.model.eval()
            act_prob = self.model(obs).cpu().numpy()
            act_prob = np.squeeze(act_prob, axis=0)  # 减少一维维度
            act = np.random.choice(range(self.num_act), p=act_prob)  # 根据动作概率选取动作
            return act
    def predict(self, obs):
        with torch.no_grad():
            obs = np.expand_dims(obs, axis = 0)
            obs = torch.tensor(obs, dtype = torch.float32).to(self.device)
            self.model.to(self.device)
            self.model.eval()
            act_prob = self.model(obs).cpu().numpy()
            act_prob = np.squeeze(act_prob, axis=0)  # 减少一维维度
            act = np.argmax(act_prob)  # 根据动作概率选择概率最高的动作
            return act
    def learn(self, obs, act, reward):
        self.model.to(self.device)
        reward = self.calc_reward_to_go(reward)
        reward = np.expand_dims(reward, axis=-1)
        act = np.expand_dims(act, axis=-1)
        obs, act, reward = torch.tensor(obs, dtype = torch.float32), torch.tensor(act, dtype = torch.int64), torch.tensor(reward, dtype = torch.float32)
        obs, act, reward = obs.to(self.device), act.to(self.device), reward.to(self.device)
        act_prob = self.model(obs)
        act_prob = torch.gather(act_prob, 1, act)
        act_prob = torch.log(act_prob)
        cost = -1 * act_prob * reward
        loss = torch.mean(cost)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    def calc_reward_to_go(self, reward_list, gamma=0.9):
        for i in range(len(reward_list) - 2, -1, -1):
            # G_t = r_t + γ·r_t+1 + ... = r_t + γ·G_t+1
            reward_list[i] += gamma * reward_list[i + 1] # Gt
        reward_list = (reward_list - np.mean(reward_list)) / np.std(reward_list)
        return reward_list
    def save(self, name):
        torch.save(self.model, os.path.join("REINFORCE", name + ".pth"))
    def load(self, path):
        self.model = torch.load(path)
        
        


        