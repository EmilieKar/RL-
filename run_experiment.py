import argparse
import gym
import importlib.util
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--agentfile", type=str, help="file with Agent object", default="sarsa.py")
parser.add_argument("--env", type=str, help="Environment", default="FrozenLake-v1")
args = parser.parse_args()

spec = importlib.util.spec_from_file_location('Agent', args.agentfile)
agentfile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agentfile)
reward = []

try:
    env = gym.make(args.env,is_slippery = False) # ev test to turn of slip , is_slippery = False
    print("Loaded ", args.env)
except:
    print(args.env +':Env')
    gym.envs.register(
        id=args.env + "-v0",
        entry_point=args.env +':Env',
    )
    env = gym.make(args.env + "-v0")
    print("Loaded", args.env)

action_dim = env.action_space.n
state_dim = env.observation_space.n

agent = agentfile.Agent(state_dim, action_dim)

observation = env.reset()

iterations = 10000
window_size = 5
stored_rewards = [0 for _ in range(window_size)]
runs = 0

for x in range(iterations): 
    #env.render()
    action = agent.act(observation) # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
    agent.observe(observation, reward, done)

    if done:
        runs += 1
        stored_rewards.append(reward)
        observation = env.reset() 

# Not the prettiest code but this plots a linechart with episodes on x-axis and 
# avg_reward on y axis with error margin indicated automatically by seaborn

values = []
for i in range(runs):
    values += stored_rewards[i:i+window_size]

episodes = [i for _ in range(window_size) for i in range(runs)]

plot_df = pd.DataFrame({'episodes': episodes, 'avg_reward': values})

print(plot_df)
    
# Plot avg_reward for 5 window
sns.lineplot(data = plot_df, x='episodes', y = 'avg_rewards')
plt.show()

# Print better overview of q
# l = left, d = down, r = right, u = up
"""
dir_dict = {'0':'l', '1': 'd', '2': 'r', '3':'u' }
dir_list = []
for a in agent.q:
    max_indx = np.where(a == np.amax(a))
    key = ""
    for elem in max_indx[0]: 
        key += dir_dict[str(elem)] 
    dir_list.append(key)


horizontal_n = 4
vertical_n = 4
for i in range(vertical_n):
    s = '|\t'
    for j in range(horizontal_n):
        tmp = dir_list[j + i*horizontal_n] 
        if len(tmp) > 3:
            tmp += '\t'
        else:
            tmp += '\t\t'
        s += tmp
    print(s + '|')
"""  
env.close()
