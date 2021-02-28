import sys
sys.path.append("REINFORCE")
import network
import torch
from torch import nn
from torch import optim
class PG_agent():
    def __init__(self, obs_dim, num_act, lr):
        self.obs_dim = obs_dim
        self.num_act = num_act 
        self.lr = lr
        self.e_greedy = e_greedy
        self.e_greedy_decrement = e_greedy_decrement
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = network.PGnetwork(self.num_obs, self.num_act)
        self.optimizer = optim.Adam(self.model.parameters(), self.lr)
    def sample(self, obs):
        with torch.no_grad():
            obs = np.expand_dims(obs, axis = 0)
            obs = torch.tensor(obs, dtype = torch.float32).to(self.device)
            self.model.to(self.device)
            self.model.eval()
            act_prob = self.model(obs).cpu()
            act_prob = np.squeeze(act_prob, axis=0)  # 减少一维维度
            act = np.random.choice(range(self.act_dim), p=act_prob)  # 根据动作概率选取动作
            return act
    def predict(self, obs):
        with torch.no_grad():
            obs = np.expand_dims(obs, axis = 0)
            obs = torch.tensor(obs, dtype = torch.float32).to(self.device)
            self.model.to(self.device)
            self.model.eval()
            act_prob = self.model(obs).cpu()
            act_prob = np.squeeze(act_prob, axis=0)  # 减少一维维度
            act = np.argmax(act_prob)  # 根据动作概率选择概率最高的动作
            return act
    def learn(self, obs, act, reward):
        self.model.to(self.device)
        self.model.train()
        act = np.expand_dims(act, axis=-1)
        reward = self.calc_reward_to_go(reward)
        obs, act, reward = torch.tensor(obs, dtype = torch.float32), torch.tensor(act, dtype = torch.int64), torch.tensor(reward, dtype = torch.float32)
        obs, act, reward = obs.to(self.device), act.to(self.device), reward.to(self.device)
        act_prob = self.model(obs)
        act_prob = torch.gather(act_prob, 1, act)
        cost = -1 * act_prob * reward
        loss = torch.mean(cost)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        
        
        


        
    def calc_reward_to_go(self, reward_list, gamma=0.9):
        for i in range(len(reward_list) - 2, -1, -1):
            # G_t = r_t + γ·r_t+1 + ... = r_t + γ·G_t+1
            reward_list[i] += gamma * reward_list[i + 1]  # Gt
        return reward_list