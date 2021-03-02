import gym
import logging
import numpy as np
import sys
sys.path.append("DDPG")
import agent
logging.basicConfig(filename="DDPG/CartPole-continuous.log")
import torch
import random
import env
import experience_replay
def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True
# 设置随机数种子
setup_seed(20)

def run_episode(env, agent, rpm):
    obs_list, action_list, reward_list = [], [], []
    obs = env.reset()
    step = 0
    total_reward = 0
    while True:
        step += 1
        obs_list.append(obs)
        action = agent.predict(obs) # 采样动作
        action = np.clip(np.random.normal(action, opt["NOISE"]), -1.0, 1.0)
        next_obs, reward, done, info = env.step(action)
        rpm.append(((obs, action, opt["REWARD_SCALE"] * reward, next_obs, done)))

        if len(rpm) > opt["MEMORY_WARMUP_SIZE"] and (step % opt["LEARN_FREQ"]) == 0:
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample(opt["BATCH_SIZE"])
            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs,
                        batch_done)

        obs = next_obs
        total_reward += reward

        if done or step >= 200:
            break
    return step, total_reward

# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(time, env, agent, render=False):
    eval_reward = []
    for i in range(time):
        obs = env.reset()
        episode_reward = 0
        step = 0
        while True:
            step += 1
            action = agent.predict(obs) # 选取最优动作
            action = np.clip(action, -1, 1)
            obs, reward, isOver, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if isOver or step >= 200:
                break
        eval_reward.append(episode_reward)
    mean_reward = np.mean(eval_reward)
    print("evaluating on {} episodes with mean reward {}.".format(time, mean_reward))
    logging.warning("evaluating on {} episodes with mean reward {}.".format(time, mean_reward))
    return mean_reward
def train(env, env_name, agent, episodes, rpm):
    while len(rpm) < opt["MEMORY_WARMUP_SIZE"]:
        run_episode(env, agent, rpm)
    for i in range(episodes):
        step, total_reward = run_episode(env, agent, rpm)
        if i % 10 == 0:
            print("Episode {}, step {} Reward Sum {}.".format(i, step, total_reward))
            logging.warning("Episode {}, step {} Reward Sum {}.".format(i, step, total_reward))

        if (i + 1) % 100 == 0:
            total_reward = evaluate(5, env, agent, render=True) 
    agent.save(env_name)

opt = {
    "LEARNING_RATE" : 1e-3,
    "ACTOR_LR" : 1e-3,  # Actor网络的 learning rate
    "CRITIC_LR" : 1e-3,  # Critic网络的 learning rate

    "GAMMA" : 0.99,      # reward 的衰减因子
    "TAU" : 0.001,       # 软更新的系数
    "MEMORY_SIZE" : int(1e6),                  # 经验池大小
    "MEMORY_WARMUP_SIZE" : int(1e6) // 200,  # 预存一部分经验之后再开始训练
    "BATCH_SIZE" : 128,
    "REWARD_SCALE" : 0.1 ,  # reward 缩放系数
    "NOISE" : 0.05,       # 动作噪声方差
    "LEARN_FREQ" : 5,
    "TRAIN_EPISODE" : 5000 # 训练的总episode数
}

if __name__ == "__main__":
    env_name = "CartPole-continous"
    env = env.ContinuousCartPoleEnv()
    print("DQN trained on {}".format(env_name))
    logging.warning("DQN trained on {}".format(env_name))
    print(opt)
    logging.warning(opt)
    act_dim = env.action_space.shape[0]
    obs_dim = env.observation_space.shape[0]
    rpm = experience_replay.ReplayMemory(opt["MEMORY_SIZE"])
    agent = agent.DDPG_agent(obs_dim = obs_dim, act_dim = act_dim, actor_lr = opt["ACTOR_LR"], critic_lr = opt["CRITIC_LR"], tau = opt["TAU"], gamma = opt["GAMMA"])
    train(env, env_name, agent, opt["TRAIN_EPISODE"], rpm)