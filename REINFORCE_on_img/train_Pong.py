import gym
import logging
import numpy as np
import sys
sys.path.append("REINFORCE_on_img")
import agent
logging.basicConfig(filename="REINFORCE_on_img/Pong-v0.log")
import torch
import random
from PIL import Image
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
    obs = preprocess(obs)
    step = 0
    while True:
        step += 1
        obs_list.append(obs)
        action = agent.sample(obs) # 采样动作
        action_list.append(action)

        obs, reward, done, info = env.step(action)
        obs = preprocess(obs)
        reward_list.append(reward)

        if done:
            break
    return obs_list, action_list, reward_list, step

# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(time, env, agent, render=False):
    eval_reward = []
    for i in range(time):
        obs = env.reset()
        obs = preprocess(obs)
        episode_reward = 0
        while True:
            action = agent.predict(obs) # 选取最优动作
            obs, reward, isOver, _ = env.step(action)
            obs = preprocess(obs)
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
        obs_list, action_list, reward_list, step = run_episode(env, agent)
        print("Episode {}, step {} Reward Sum {}.".format(i, step, sum(reward_list)))
        logging.warning("Episode {}, step {} Reward Sum {}.".format(i, step, sum(reward_list)))
        batch_obs = obs_list
        batch_action = np.array(action_list)
        batch_reward = np.array(reward_list)

        agent.learn(batch_obs, batch_action, batch_reward)
        if (i + 1) % 100 == 0:
            total_reward = evaluate(5, env, agent, render=False) 
    agent.save(env_name)

opt = {
    "LEARNING_RATE" : 1e-3
}


def preprocess(image):
    """ 预处理 210x160x3 uint8 frame into 6400 (80x80) 1维 float vector """
    image = image[35:195] # 裁剪
    image = image[::2,::2, 0] # 下采样，缩放2倍
    image[image == 144] = 0 # 擦除背景 (background type 1)
    image[image == 109] = 0 # 擦除背景 (background type 2)
    image[image != 0] = 1 # 转为灰度图，除了黑色外其他都是白色
    image = np.expand_dims(image, 0)
    return image




if __name__ == "__main__":
    env_name = "Pong-v0"
    env = gym.make(env_name)
    print("DQN trained on {}".format(env_name))
    logging.warning("DQN trained on {}".format(env_name))
    print(opt)
    logging.warning(opt)
    num_act = env.action_space.n
    obs_shape = (80, 80, 1)
    agent = agent.PG_agent(obs_shape, num_act, opt["LEARNING_RATE"])
    train(env, env_name, agent, 3000)
