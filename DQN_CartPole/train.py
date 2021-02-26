import gym
import sys
sys.path.append("DQN_CartPole")
import agent
import Q_network
import experience_replay
import torch
import numpy as np
import logging
logging.basicConfig(filename = "DQN_CartPole/DQN_CartPole.log")
def run_episode(env, agent, rpm):
    total_reward = 0
    obs = env.reset()
    step = 0
    while True:
        step += 1
        action = agent.sample(obs) 
        next_obs, reward, done, _ = env.step(action)
        rpm.append((obs, action, reward, next_obs, done))

        # train model
        if (len(rpm) > opt["MEMORY_WARMUP_SIZE"] and (step % opt["LEARN_FREQ"] == 0)):
            (batch_obs, batch_action, batch_reward, batch_next_obs, batch_done) = rpm.sample(opt["BATCH_SIZE"])
            agent.learn(batch_obs, batch_action, batch_reward, batch_next_obs, batch_done)  # s,a,r,s',done

        total_reward += reward
        obs = next_obs
        if done:
            break
    return total_reward, step


# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent, render=False):
    with torch.no_grad():
        eval_reward = []
        for i in range(5):
            obs = env.reset()
            episode_reward = 0
            while True:
                action = agent.predict(obs)  # 预测动作，只选最优动作
                obs, reward, done, _ = env.step(action)
                episode_reward += reward
                if render:
                    env.render()
                if done:
                    break
            eval_reward.append(episode_reward)
    return np.mean(eval_reward)
def train(episodes, env, agent, save):
    rpm = experience_replay.ReplayMemory(opt["MEMORY_SIZE"])
    while len(rpm) < opt["MEMORY_WARMUP_SIZE"]:
        run_episode(env, agent, rpm)
    for episode in range(episodes):
        reward, steps = run_episode(env, agent, rpm)
        # reward, steps = run_episode_with_sarsa(env, agent, False)
        print("train episode {} : reward {}, steps {}".format(episode + 1, reward, steps))
        logging.warning("train episode {} : reward {}, steps {}".format(episode + 1, reward, steps))
        if episode % 50 == 0:
            eval_reward = evaluate(env, agent, render = True)
            print("evaluate 5 episodes : e_greedy {}, reward {}".format(agent.e_greedy, eval_reward))
            logging.warning("evaluate 5 episodes : e_greedy {}, reward {}".format(agent.e_greedy, eval_reward))
    if save:
        agent.save()
    return agent

opt = {
    "LEARN_FREQ" : 5, # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
    "MEMORY_SIZE" : 20000,    # replay memory的大小，越大越占用内存
    "MEMORY_WARMUP_SIZE" : 200,  # replay_memory 里需要预存一些经验数据，再开启训练
    "BATCH_SIZE" : 32,   # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
    "LEARNING_RATE" : 0.001, # 学习率
    "GAMMA" : 0.99, # reward 的衰减因子，一般取 0.9 到 0.999 不等
    "E_GREEDY" : 0.1,
    "E_GREEDY_DECREMENT" : 1e-6,
    "max_episode" : 2000
}

if __name__ == "__main__":
    env_name = "CartPole-v0"
    env = gym.make(env_name)
    logging.warning("DQN trained on {}".format(env_name))
    logging.warning("configuration:")
    logging.warning(opt)
    num_act = env.action_space.n
    num_obs = env.observation_space.shape[0]
    dqn_agent = agent.DQN_agent(num_act, num_obs, opt["GAMMA"], opt["LEARNING_RATE"], opt["E_GREEDY"], opt["E_GREEDY_DECREMENT"])
    train(opt["max_episode"], env, dqn_agent, True)