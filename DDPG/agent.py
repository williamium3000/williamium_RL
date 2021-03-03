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
        self.actor = network.Actor(obs_dim, act_dim)
        self.critic = network.Critic(obs_dim, act_dim)
        self.target_actor = network.Actor(obs_dim, act_dim)
        self.target_critic = network.Critic(obs_dim, act_dim)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)
        self.actor_optim = optim.Adam(self.actor.parameters(), self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), self.critic_lr)
        self.global_step = 0
        self.update_target_steps = 1
    def predict(self, obs):
        with torch.no_grad():
            self.actor.to(self.device)
            obs = np.expand_dims(obs, axis = 0)
            obs = torch.tensor(obs, dtype=torch.float32).to(self.device)
            return self.actor(obs).detach().cpu().numpy()[0]

    def learn(self, state, action, reward, next_state, done):
        self.global_step += 1
        self.actor.to(self.device)
        self.critic.to(self.device)
        self.target_actor.to(self.device)
        self.target_critic.to(self.device)
        # # print("obs.shape {}".format(obs.shape))

        # terminal = np.expand_dims(terminal, axis = -1)
        # reward = np.expand_dims(reward, axis = -1)
        
        # obs, act, reward, next_obs, terminal = torch.tensor(obs, dtype = torch.float32), torch.tensor(action, dtype = torch.float32), torch.tensor(reward, dtype = torch.float32), torch.tensor(next_obs, dtype = torch.float32), torch.tensor(terminal, dtype = torch.float32)
        # obs, act, reward, next_obs, terminal = obs.to(self.device), act.to(self.device), reward.to(self.device), next_obs.to(self.device), terminal.to(self.device)
        
        # action = self.actor(obs)
        # # print("action.shape {}".format(action.shape))
        # obs_and_act = torch.cat([obs, action], dim = -1)
        # # print("obs_and_act.shape {}".format(obs_and_act.shape))
        # Q = self.target_critic(obs_and_act)
        # # print("Q.shape {}".format(Q.shape))
        # loss = torch.mean(-1.0 * Q)
        # self.actor_optim.zero_grad()
        # loss.backward()
        # self.actor_optim.step()


        # # print("obs.shape {}".format(obs.shape))
        # # print("act.shape {}".format(act.shape))
        # # print("reward.shape {}".format(reward.shape))
        # # print("next_obs.shape {}".format(next_obs.shape))
        # # print("terminal.shape {}".format(terminal.shape))

        # next_action = self.target_actor(next_obs)
        # # print("next_action.shape {}".format(next_action.shape))
        # obs_and_act = torch.cat([next_obs, next_action], dim = -1)
        # # print("obs_and_act.shape {}".format(obs_and_act.shape))
        # next_Q = self.target_critic(obs_and_act)
        # target_Q = reward + (1.0 - terminal) * self.gamma * next_Q
        # # print("target_Q.shape {}".format(target_Q.shape))

        # obs_and_act2 = torch.cat([obs, act], dim = -1) 
        # # print("obs_and_act2.shape {}".format(obs_and_act2.shape))
        # Q = self.critic(obs_and_act2)
        # # print("Q.shape {}".format(Q.shape))
        # loss = nn.MSELoss()(Q, target_Q)
        # self.critic_optim.zero_grad()
        # loss.backward()
        # self.critic_optim.step()


        # 将所有变量转为张量
        state = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action = torch.FloatTensor(action).to(self.device)
        reward = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        # 注意critic将(s_t,a)作为输入
        policy_loss = self.critic(state, self.actor(state))
        
        policy_loss = -policy_loss.mean()

        next_action = self.target_actor(next_state)
        target_value = self.target_critic(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * self.gamma * target_value
        expected_value = torch.clamp(expected_value, -np.inf, np.inf)

        value = self.critic(state, action)
        value_loss = nn.MSELoss()(value, expected_value.detach())
        
        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        self.critic_optim.zero_grad()
        value_loss.backward()
        self.critic_optim.step()
        




        if self.global_step % self.update_target_steps == 0:
            self.sync_target()


    def sync_target(self, decay=None, share_vars_parallel_executor=None):
        """ self.target_model从self.model复制参数过来，可设置软更新参数
        """
        if decay is None:
            decay = self.tau
        self.target_actor.to("cpu")
        self.target_critic.to("cpu")
        self.actor.to("cpu")
        self.critic.to("cpu")
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    # def save(self, name):
    #     torch.save(self.model, os.path.join("DDPG", name + ".pth"))
    # def load(self, path):
    #     self.model = torch.load(path)
    #     self.sync_target()
