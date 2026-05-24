import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque

class DQNAgent:
    """
    Deep Q-Network (DQN) Agent representing the decision maker in RLSFLoc.
    """
    def __init__(self, state_dim, action_dim=2, lr=0.001, gamma=0.99):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        
        # Q-Network
        self.q_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, action_dim)
        )
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=2000)

    def select_action(self, state, epsilon=0.1):
        """
        Select action using epsilon-greedy strategy.
        """
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_step(self, batch_size=32):
        if len(self.memory) < batch_size:
            return 0.0
            
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.LongTensor(actions).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_t = torch.FloatTensor(np.array(next_states))
        dones_t = torch.FloatTensor(dones).unsqueeze(1)
        
        current_q = self.q_net(states_t).gather(1, actions_t)
        
        with torch.no_grad():
            max_next_q = torch.max(self.q_net(next_states_t), dim=1, keepdim=True)[0]
            target_q = rewards_t + (self.gamma * max_next_q * (1 - dones_t))
            
        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
