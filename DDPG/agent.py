import sys
sys.path.append("DDPG")
import network
import copy
import torch
import numpy as np
from torch import nn, optim
import os
class DDPG_agent():
    def __init__(self, obs_dim, act_dim, actor_lr, critic_lr, tau, gamma):
        self.gamma = gamma
        self.tau = tau
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = network.DDPGnetwork(obs_dim, act_dim)
        self.target_model = network.DDPGnetwork(obs_dim, act_dim)
        self.target_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        self.actor_optim = optim.Adam(self.model.actor.parameters(), self.actor_lr)
        self.critic_optim = optim.Adam(self.model.critic.parameters(), self.critic_lr)
        self.global_step = 0
        self.update_target_steps = 500
    def predict(self, obs):
        with torch.no_grad():
            self.model.to(self.device)
            obs = np.expand_dims(obs, axis = 0)
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
            return self.model(obs).cpu().numpy()[0]

    def learn(self, obs, action, reward, next_obs, terminal):
        self.global_step += 1
        if self.global_step % self.update_target_steps == 0:
            self.sync_target()
        actor_cost = self._actor_learn(obs)
        critic_cost = self._critic_learn(obs, action, reward, next_obs,
                                         terminal)
        return actor_cost, critic_cost

    def _actor_learn(self, obs):
        self.model.to(self.device)
        self.model.train()
        obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
        action = self.model(obs)
        Q = self.model.value(obs, action)
        loss = torch.mean(-1.0 * Q)
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()

    def _critic_learn(self, obs, act, reward, next_obs, terminal):
        self.model.to(self.device)
        self.model.train()
        terminal = np.expand_dims(terminal, axis = 0)
        reward = np.expand_dims(reward, axis = 0)
        obs, act, reward, next_obs, terminal = torch.tensor(obs, dtype = torch.float32), torch.tensor(act, dtype = torch.float32), torch.tensor(reward, dtype = torch.float32), torch.tensor(next_obs, dtype = torch.float32), torch.tensor(terminal, dtype = torch.float32)
        obs, act, reward, next_obs, terminal = obs.to(self.device), act.to(self.device), reward.to(self.device), next_obs.to(self.device), terminal.to(self.device)
        self.target_model.to(self.device)
        self.target_model.eval()
        with torch.no_grad():
            next_action = self.target_model(next_obs)
            next_Q = self.target_model.value(next_obs, next_action)
            target_Q = reward + (1.0 - terminal) * self.gamma * next_Q

        Q = self.model.value(obs, act)
        loss = nn.MSELoss()(Q, target_Q)
        self.actor_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()


    def sync_target(self, decay=None, share_vars_parallel_executor=None):
        """ self.target_model从self.model复制参数过来，可设置软更新参数
        """
        if decay is None:
            decay = self.tau
        d1 = self.target_model.state_dict()
        d2 = self.model.state_dict()
        for key, value in d2.items():
            d1[key] = decay * d2[key] + (1 - decay) * d1[key]
        self.target_model.load_state_dict(d1)
        self.target_model.eval()
    def save(self, name):
        torch.save(self.model, os.path.join("DDPG", name + ".pth"))
    def load(self, path):
        self.model = torch.load(path)
        self.sync_target()
