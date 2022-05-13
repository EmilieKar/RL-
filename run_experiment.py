import argparse
import gym
import importlib.util
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--agentfile", type=str, help="file with Agent object", default="doubleq.py")
parser.add_argument("--env", type=str, help="Environment", default="FrozenLake-v1")
args = parser.parse_args()

spec = importlib.util.spec_from_file_location('Agent', args.agentfile)
agentfile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(agentfile)
reward = []

try:
    env = gym.make(args.env,is_slippery = True) # ev test to turn of slip , is_slippery = False
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

iterations = 30000

# Helper vars for moving average plot
window_size = 100
stored_rewards = [0 for _ in range(window_size)]

for x in range(iterations): 
    #env.render()
    action = agent.act(observation) # your agent here (this takes random actions)
    observation, reward, done, info = env.step(action)
    agent.observe(observation, reward, done)

    if done:
        observation = env.reset() 
    
    stored_rewards.append(reward)

# Plot calculateions
values = []
st_devs = []
sqrt_window = window_size ** (1/2)

for i in range(iterations):
    tmp = stored_rewards[i:i+window_size]
    values.append(sum(tmp)/window_size)
    st_devs.append((sum([((x - values[-1]) ** 2) for x in tmp]) / window_size) ** 0.5)

ci = [1.96 * (st_devs[i]/sqrt_window) for i in range(iterations)]

plot_df = pd.DataFrame({
    'avg_reward': values,
    'high_ci': [values[i] + ci[i] for i in range(iterations)],
    'low_ci':[values[i] - ci[i] for i in range(iterations)]
})
    
# Plot avg_reward for moving window
p = sns.lineplot(data = plot_df, x=plot_df.index, y = 'avg_reward', ci=None)
p.fill_between(plot_df.index, plot_df.low_ci, plot_df.high_ci, alpha=0.2)
plt.xlabel('iterations')
plt.title('Average reward (doubleq-learning, FroznLake-v1, is_slippery=True)')
plt.show()

# Print better overview of q
# l = left, d = down, r = right, u = up
dir_dict = {'0':'l', '1': 'd', '2': 'r', '3':'u' }
dir_list = []

# Variable to indicate grid size
horizontal_n = 4
vertical_n = 4

# Map numerical values to letters
for a in agent.q:
    max_indx = np.where(a == np.amax(a))
    key = ""
    for elem in max_indx[0]: 
        key += dir_dict[str(elem)] 
    dir_list.append(key)

# Print actions as grid
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
 
env.close()
