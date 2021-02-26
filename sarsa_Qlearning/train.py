import gym
import sys
sys.path.append("sarsa_Qlearning")
import agent
import gridworld
import time
def run_episode_with_sarsa(env, agent, render = False):
    steps = 0
    total_reward = 0
    state = env.reset()
    action = agent.sample(state)
    while True:
        next_state, reward, done, _ = env.step(action) # 与环境进行一个交互
        next_action = agent.sample(next_state) # 根据算法选择一个动作
        # train Sarsa
        agent.learn(state, action, reward, next_state, next_action, done)
        action = next_action
        state = next_state  # 存储上一个观察值
        total_reward += reward
        steps += 1 # 计算step数
        if render:
            env.render() #渲染新的一帧图形
        if done:
            break
    return total_reward, steps

def run_episode_with_Qlearning(env, agent, render = False):
    steps = 0
    total_reward = 0
    state = env.reset()
    while True:
        action = agent.sample(state)
        next_state, reward, done, _ = env.step(action) # 与环境进行一个交互
        # train Q-learning
        agent.learn(state, action, reward, next_state, done)
        state = next_state  # 存储上一个观察值
        total_reward += reward
        steps += 1 # 计算step数
        if render:
            env.render() #渲染新的一帧图形
        if done:
            break
    return total_reward, steps
 
def test_episode(env, agent):
    steps = 0
    total_reward = 0
    state = env.reset()
    while True:
        time.sleep(0.5)
        action = agent.predict(state) # 根据算法选择一个动作
        state, reward, done, _ = env.step(action) # 与环境进行一个交互
        total_reward += reward
        steps += 1 # 计算step数
        env.render()
        if done:
            break
    return total_reward, steps

def train(episodes, env, agent, save):
    for episode in range(episodes):
        reward, steps = run_episode_with_Qlearning(env, agent, False)
        # reward, steps = run_episode_with_sarsa(env, agent, False)
        print("episode {} : reward {}, steps {}".format(episode + 1, reward, steps))
    if save:
        agent.save()
    return agent
def test(agent, env):
    reward, steps = test_episode(env, agent)
    print("test on env : reward {}, steps {}".format(reward, steps))


if __name__ == "__main__":
    # env = gym.make("CliffWalking-v0")
    env = gym.make("FrozenLake-v0", is_slippery = True)
    env = gridworld.FrozenLakeWapper(env)
    # sarsa_agent = agent.sarsaAgent(
    #     num_state=env.observation_space.n,
    #     num_act=env.action_space.n,
    #     lr=0.1,
    #     gamma=0.95,
    #     e_greedy=0.1)
    QLearning_agent = agent.QLearningAgent(
        num_state=env.observation_space.n,
        num_act=env.action_space.n,
        lr=0.1,
        gamma=0.9,
        e_greedy=0.1)
    # agent = train(5000, env, QLearning_agent, True)
    QLearning_agent.restore("sarsa_Qlearning/q_table.npy")
    test(QLearning_agent, env)

