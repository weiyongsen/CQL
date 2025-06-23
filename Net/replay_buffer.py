import numpy as np
import torch
from collections import deque
import random

class ReplayBuffer:
    def __init__(self, max_size=int(1e6)):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
    
    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return (torch.FloatTensor(np.array(state)),
                torch.FloatTensor(np.array(action)),
                torch.FloatTensor(np.array(reward)).unsqueeze(1),
                torch.FloatTensor(np.array(next_state)),
                torch.FloatTensor(np.array(done)).unsqueeze(1))
    
    def __len__(self):
        return len(self.buffer) 