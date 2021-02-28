import gym
import logging
import numpy as np
logging.basicConfig(filename="REINFORCE/CartPole-v0.log")
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
def train(env, agent, episodes):
    for i in range(episodes):
        obs_list, action_list, reward_list = run_episode(env, agent)
        if i % 10 == 0:
            logger.info("Episode {}, Reward Sum {}.".format(i, sum(reward_list)))

        batch_obs = np.array(obs_list)
        batch_action = np.array(action_list)
        batch_reward = np.array(reward_list)

        agent.learn(batch_obs, batch_action, batch_reward)
        if (i + 1) % 100 == 0:
            total_reward = evaluate(5, env, agent, render=True) 

if __name__ == "__main__":
    env_name = "CartPole-v0"
    env = gym.make(env_name)
    logging.warning("DQN trained on {}".format(env_name))
    logging.warning(opt)
    num_act = env.action_space.n
    num_obs = env.observation_space.shape[0]
