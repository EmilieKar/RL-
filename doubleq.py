from re import L
import numpy as np

class Agent(object): #Keep the class name!
    """The world's simplest agent!"""
    def __init__(self, state_space, action_space):
        self.action_space = action_space
        self.state_space = state_space
        self.q = np.zeros((self.state_space,self.action_space))
        self.q1 = np.zeros((self.state_space,self.action_space))
        self.q2 = np.zeros((self.state_space,self.action_space))
        self.a = 0.1
        self.g = 0.85
        self.e = 0.05
        self.action = 0
        self.state = 0
        self.rewards = 0
    
    def observe(self, observation, reward, done):
        if np.random.uniform(0,1) > 0.5:
            self.q1[self.state,self.action] = self.q1[self.state,self.action] + self.a*(reward + self.g*self.q2[observation,np.random.choice(np.flatnonzero(self.q1[observation] == max(self.q1[observation])))] - self.q1[self.state,self.action])
        else:
            self.q2[self.state,self.action] = self.q2[self.state,self.action] + self.a*(reward + self.g*self.q1[observation,np.random.choice(np.flatnonzero(self.q2[observation] == max(self.q2[observation])))] - self.q2[self.state,self.action])
        self.rewards += reward
        self.q = np.add(self.q1,self.q2)
        
            
    def act(self, observation):
        #Add your code here
        if np.random.uniform(0,1) > self.e:
            self.action = np.random.choice(np.flatnonzero(self.q[observation] == max(self.q[observation])))
        else:
            self.action = np.random.randint(self.action_space)
        
        self.state = observation
        return self.action