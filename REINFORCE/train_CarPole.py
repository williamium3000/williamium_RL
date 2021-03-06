import gym
import logging
import numpy as np
import sys
sys.path.append("REINFORCE")
import agent

import torch
import random
import numpy as np
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)

def run_episode(env, agent):
    obs_list, action_list, reward_list = [], [], []
    obs = env.reset()
    while True:
        obs_list.append(obs)
        action = agent.sample(obs) # 采样动作
        action_list.append(action)

        obs, reward, done, info = env.step(action)
        reward_list.append(reward)

        if done:
            break
    return obs_list, action_list, reward_list

# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(time, env, agent, render=False):
    eval_reward = []
    for i in range(time):
        obs = env.reset()
        episode_reward = 0
        while True:
            action = agent.predict(obs) # 选取最优动作
            obs, reward, isOver, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if isOver:
                break
        eval_reward.append(episode_reward)
    mean_reward = np.mean(eval_reward)
    print("evaluating on {} episodes with mean reward {}.".format(time, mean_reward))
    logging.warning("evaluating on {} episodes with mean reward {}.".format(time, mean_reward))
    return mean_reward
def train(env, env_name, agent, episodes):
    for i in range(episodes):
        obs_list, action_list, reward_list = run_episode(env, agent)
        if i % 10 == 0:
            print("Episode {}, Reward Sum {}.".format(i, sum(reward_list)))
            logging.warning("Episode {}, Reward Sum {}.".format(i, sum(reward_list)))


        batch_obs = np.array(obs_list)
        batch_action = np.array(action_list)
        batch_reward = np.array(reward_list)

        agent.learn(batch_obs, batch_action, batch_reward)
        if (i + 1) % 100 == 0:
            total_reward = evaluate(5, env, agent, render=False) 
    agent.save(env_name)

opt = {
    "LEARNING_RATE" : 0.001
}

if __name__ == "__main__":
    env_name = "CartPole-v0"
    env = gym.make(env_name)
    logging.basicConfig(filename="REINFORCE/{}.log".format(env_name))

    print("DQN trained on {}".format(env_name))
    logging.warning("DQN trained on {}".format(env_name))
    print(opt)
    logging.warning(opt)
    num_act = env.action_space.n
    obs_dim = env.observation_space.shape[0]
    agent = agent.PG_agent(obs_dim, num_act, opt["LEARNING_RATE"])
    train(env, env_name, agent, 500)
