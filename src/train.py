import numpy as np
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
from tqdm import tqdm
import torch 
import torch.nn as nn 
import random
from copy import deepcopy
#import matplotlib.pyplot as plt 
import pickle 

from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
    

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!
class ProjectAgent:
    def __init__(self):
        self.nb_actions=4
        self.memory=ReplayBuffer(10000)
        #self.init_buffer()
        #print('Buffer initialized')
        self.nb_iter=10
        self.max_episode=10
        #self.Qfunctions = self.fqi(0.98)

    def act(self, observation, use_random=False):
        action=self.greedy_action(self.Qfunctions[-1], observation)
        return action 
    
    def save(self, path):
        path=path+"model.pickle"
        pickle.dump(self.Qfunctions, open(path, "wb"))
        return
    
    def load(self): 
        path="tree/model.pickle"
        self.Qfunctions = pickle.load(open(path, "rb"))
        return

    def fqi(self, gamma):
        self.Qfunctions = []
        state, _ = env.reset()
        step = 0
        epsilon_min=0.1
        epsilon_max=1.
        epsilon_stop=100
        epsilon_delay=100
        epsilon = epsilon_max
        epsilon_step = (epsilon_max-epsilon_min)/epsilon_stop
        batch_size=100
        episode_return = []
        episode=0
        episode_cum_reward = 0
        while episode < self.max_episode:
            if step > epsilon_delay:
                epsilon = max(epsilon_min, epsilon-epsilon_step)
            if len(self.memory) < batch_size:
                action = np.random.randint(self.nb_actions)
            else : 
                if np.random.rand() < epsilon:
                    action = np.random.randint(self.nb_actions)
                else: 
                    S, A, R, S2, D = self.memory.sample_memory()
                    nb_samples = S.shape[0]
                    SA = np.append(S,A,axis=1)
                    for iter in range(self.nb_iter):
                        if iter==0 & episode==0:
                            value=R.copy()
                        else: 
                            Q2 = np.zeros((nb_samples,self.nb_actions))
                            for a2 in range(self.nb_actions):
                                A2 = a2*np.ones((S.shape[0],1))
                                S2A2 = np.append(S2,A2,axis=1)
                                Q2[:,a2] = (self.Qfunctions[-1]).predict(S2A2)
                            max_Q2 = np.max(Q2,axis=1)
                            value = R + gamma*(1-D)*max_Q2
                        Q = HistGradientBoostingRegressor()
                        Q.fit(SA,value)
                        self.Qfunctions.append(Q)
                        if len(self.Qfunctions)>1:
                            self.Qfunctions.pop(0)

                    action = self.greedy_action(self.Qfunctions[-1], state)
            step=step+1
             # step
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)
            episode_cum_reward += reward

            if done or trunc:
                episode += 1
                print("Episode ", '{:3d}'.format(episode), 
                    ", epsilon ", '{:6.2f}'.format(epsilon), 
                    ", batch size ", '{:5d}'.format(len(self.memory)), 
                    ", episode return ", '{:4.1f}'.format(episode_cum_reward),
                        sep='')
                state, _ = env.reset()
                episode_return.append(episode_cum_reward)
                episode_cum_reward = 0
            else:
                state = next_state

        self.save('tree/')
        #plt.plot(episode_return)
        #plt.show()
        return self.Qfunctions
    
      
    def greedy_action(self,Q, s):
        Qsa = []
        for a in range(self.nb_actions):
            sa = np.append(s,a).reshape(1, -1)
            Qsa.append(Q.predict(sa))
        return np.argmax(Qsa)
    
    def init_buffer(self):
        state, _ = env.reset()
        for i in tqdm(range(self.memory.capacity)): 
            action = np.random.randint(self.nb_actions)
            next_state, reward, done, trunc, _ = env.step(action)
            self.memory.append(state, action, reward, next_state, done)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = int(capacity) # capacity of the buffer
        self.S = []
        self.A = []
        self.R = []
        self.S2 = []
        self.D = []
        self.index = 0 # index of the next cell to be filled

    def append(self, s, a, r, s_, d):
        if len(self.S) > self.capacity:
            self.S.pop(0)
            self.A.pop(0)
            self.R.pop(0)
            self.S2.pop(0)
            self.D.pop(0)
        self.S.append(s)
        self.A.append(a)
        self.R.append(r)
        self.S2.append(s_)
        self.D.append(d)
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size):
        batch_index = random.sample(range(len(self.S)), batch_size)
        S_batch=[self.S[ind] for ind in batch_index]
        A_batch=[self.A[ind] for ind in batch_index]
        R_batch=[self.R[ind] for ind in batch_index]
        S2_batch=[self.S2[ind] for ind in batch_index]
        D_batch=[self.D[ind] for ind in batch_index]
        S_batch = np.array(S_batch)
        A_batch = np.array(A_batch).reshape((-1,1))
        R_batch = np.array(R_batch)
        S2_batch= np.array(S2_batch)
        D_batch = np.array(D_batch)
        return S_batch, A_batch, R_batch, S2_batch, D_batch
    
    def sample_memory(self):
        S_batch = np.array(self.S)
        A_batch = np.array(self.A).reshape((-1,1))
        R_batch = np.array(self.R)
        S2_batch= np.array(self.S2)
        D_batch = np.array(self.D)
        return S_batch, A_batch, R_batch, S2_batch, D_batch
    
    def __len__(self):
        return self.index
