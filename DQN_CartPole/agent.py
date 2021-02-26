import sys
sys.path.append("DQN_CartPole")
import Q_network
import numpy as np
import torch
from torch import nn
import copy
class DQN_agent():
    def __init__(self, num_act, dim_obs, gamma, lr, e_greedy, e_greed_decrement):
        self.model = Q_network.Q_network(dim_obs, num_act)
        self.target_model = Q_network.Q_network(dim_obs, num_act)
        self.target_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        self.Loss = nn.MSELoss()
        self.optim = torch.optim.Adam(self.model.parameters(), lr)


        self.dim_obs = dim_obs
        self.num_act = num_act
        self.gamma = gamma
        self.lr = lr
        self.global_step = 0
        self.update_target_steps = 200  # 每隔200个training steps再把model的参数复制到target_model中

        self.e_greedy = e_greedy  # 有一定概率随机选取动作，探索
        self.e_greedy_decrement = e_greed_decrement  # 随着训练逐步收敛，探索的程度慢慢降低
    def sample(self, obs):
        sample = np.random.rand()
        if sample < self.e_greedy:
            action = np.random.choice(self.num_act)
        else:
            action = self.predict(obs)
        self.e_greedy = max(0.01, self.e_greedy - self.e_greedy_decrement)
        return action
    def predict(self, obs):
        self.model.eval()
        obs = np.expand_dims(obs, axis = 0)
        obs = torch.tensor(obs, dtype = torch.float32)
        with torch.no_grad():
            pred_Q = self.model(obs)
            pred_Q = np.squeeze(pred_Q, axis=0)
            act = np.argmax(pred_Q).item()  # 选择Q最大的下标，即对应的动作
        return act
    def learn(self, obs, act, reward, next_obs, terminal):
        self.model.train()
        if self.global_step % self.update_target_steps == 0:
            self.sync_target()
        self.global_step += 1
        act = np.expand_dims(act, -1)
        obs, act, reward, next_obs, terminal = torch.tensor(obs, dtype = torch.float32), torch.tensor(act, dtype = torch.int64), torch.tensor(reward, dtype = torch.float32), torch.tensor(next_obs, dtype = torch.float32), torch.tensor(terminal, dtype = torch.float32)
        with torch.no_grad():
            next_pred_value = self.target_model(next_obs)
            best_value = torch.max(next_pred_value, -1)[0]
            target = reward + (1.0 - terminal) * self.gamma * best_value
        y = self.model(obs)
        y = torch.gather(y, 1, act)
        loss = self.Loss(y, target.reshape(target.shape[0], -1))
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        



    def save(self):
        torch.save(self.model, "DQN_CartPole/Q_network.pth")
    def load(self, path):
        self.model = torch.load(open(path, 'r'))
        self.sync_target()

    def sync_target(self):
        self.target_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        self.target_model.eval()

    