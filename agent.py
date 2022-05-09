# q-learning
from re import L
import numpy as np

class Agent(object): #Keep the class name!
    """The world's simplest agent!"""
    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.state_space = state_space
        self.q = np.zeros((self.state_space,self.action_space))
        self.a = 0.1
        self.g = 0.5
        self.e = 0.05
        self.action = 0
        self.state = 0
        self.rewards = 0
    
    def observe(self, observation, reward, done):
        #print(self.state,self.action)
        # TODO fix done
        if done:
            self.q[self.state,self.action] = self.q[self.state,self.action] + self.a*(reward - self.q[self.state,self.action])
        else:
            self.q[self.state,self.action] = self.q[self.state,self.action] + self.a*(reward + self.g*np.max(self.q[observation]) - self.q[self.state,self.action])
        
        self.rewards += reward
        

    def act(self, observation):
        #Add your code here
        if np.random.uniform(0,1) > self.e:
            self.action = np.random.choice(np.flatnonzero(self.q[observation] == max(self.q[observation])))
        else:
            self.action = np.random.randint(self.action_space)
        
        self.state = observation
        return self.action