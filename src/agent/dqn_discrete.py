import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

def generate_simplex_grid(step_size=0.1):
    """
    Generate all unique continuous weight combinations on the 3D simplex.
    Satisfies sum(lambda) == 1.0 and lambda >= 0.0.
    For step_size = 0.1, this returns 66 unique combinations.
    """
    grid = []
    M = int(round(1.0 / step_size))
    for i in range(M + 1):
        for j in range(M + 1 - i):
            k = M - i - j
            grid.append([i / M, j / M, k / M])
    return np.array(grid, dtype=np.float32)

class SimplexDQNAgent:
    """
    Deep Q-Network (DQN) agent with Simplex Discretization for RLSFLoc.
    Discretizes the continuous 3D simplex weight space into a finite set of grid actions.
    Uses log-transform to invert the environment's Softmax simplex projection, ensuring
    perfect application of selected weights.
    """
    def __init__(self, state_dim=9, step_size=0.1, lr=0.001, gamma=0.99, memory_size=2000):
        self.simplex_grid = generate_simplex_grid(step_size)
        self.action_dim = len(self.simplex_grid)
        self.gamma = gamma
        
        # Q-Network
        self.q_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()
        self.memory = deque(maxlen=memory_size)

    def select_action(self, state, epsilon=0.1):
        """
        Select action index using epsilon-greedy policy.
        Returns:
          - action_idx: discrete action index (0 to action_dim-1)
          - continuous_action: continuous continuous 3D action array to be passed to environment
            (log transformed to bypass the environment's internal Softmax projection)
          - simplex_weights: actual selected simplex weights [lambda1, lambda2, lambda3]
        """
        if random.random() < epsilon:
            action_idx = random.randint(0, self.action_dim - 1)
        else:
            state_t = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                q_values = self.q_net(state_t)
            action_idx = int(torch.argmax(q_values).item())
            
        simplex_weights = self.simplex_grid[action_idx]
        
        # Log-transform to invert the environment's internal Softmax function
        # Softmax(log(w + eps)) yields exactly w.
        continuous_action = np.log(simplex_weights + 1e-12)
        
        return action_idx, continuous_action, simplex_weights

    def remember(self, state, action_idx, reward, next_state, done):
        self.memory.append((state, action_idx, reward, next_state, done))

    def train_step(self, batch_size=32):
        """
        Sample a batch from memory and optimize Q-Network weights using TD-loss.
        """
        if len(self.memory) < batch_size:
            return 0.0
            
        batch = random.sample(self.memory, batch_size)
        states, action_indices, rewards, next_states, dones = zip(*batch)
        
        states_t = torch.FloatTensor(np.array(states))
        action_indices_t = torch.LongTensor(action_indices).unsqueeze(1)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_t = torch.FloatTensor(np.array(next_states))
        dones_t = torch.FloatTensor(dones).unsqueeze(1)
        
        # Compute current Q-values
        current_q = self.q_net(states_t).gather(1, action_indices_t)
        
        # Compute target Q-values
        with torch.no_grad():
            max_next_q = torch.max(self.q_net(next_states_t), dim=1, keepdim=True)[0]
            target_q = rewards_t + (self.gamma * max_next_q * (1.0 - dones_t))
            
        loss = self.loss_fn(current_q, target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
