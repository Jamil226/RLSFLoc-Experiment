import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

class ActorCriticAgent:
    """
    Continuous Advantage Actor-Critic (A2C) Agent for RLSFLoc.
    Actor predicts continuous action distributions, and Critic estimates state values
    to compute temporal-difference advantages for stable gradient steps.
    """
    def __init__(self, state_dim=9, action_dim=3, lr=0.002, gamma=0.99, entropy_coef=0.01, value_coef=0.5):
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        # Actor network: output mean mu
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim)
        )
        
        # Trainable log std parameter
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic network: output state value V(s)
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr},
            {'params': [self.log_std], 'lr': lr},
            {'params': self.critic.parameters(), 'lr': lr}
        ])
        
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        """
        Sample continuous 3D action from Gaussian policy.
        Returns: action array, log_prob, state value estimate.
        """
        state_t = torch.FloatTensor(state)
        
        with torch.no_grad():
            mu = self.actor(state_t)
            std = torch.exp(self.log_std)
            dist = Normal(mu, std)
            
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)
            value = self.critic(state_t)
            
        return action.numpy(), log_prob.item(), value.item()

    def update(self, states, actions, log_probs, returns):
        """
        Update Actor and Critic networks using temporal difference advantage.
        """
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.FloatTensor(np.array(actions))
        log_probs_t = torch.FloatTensor(np.array(log_probs))
        returns_t = torch.FloatTensor(np.array(returns))
        
        # Forward passes
        mu = self.actor(states_t)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)
        
        # Evaluate current actions
        new_log_probs = dist.log_prob(actions_t).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        values = self.critic(states_t).squeeze(-1)
        
        # TD Advantage: G_t - V(s_t)
        advantages = returns_t - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Actor loss (policy gradient with baseline and entropy)
        actor_loss = -(new_log_probs * advantages).mean()
        # Critic loss (value regression)
        critic_loss = self.loss_fn(values, returns_t)
        # Entropy regularizer (exploration encouragement)
        entropy_loss = -self.entropy_coef * entropy.mean()
        
        total_loss = actor_loss + self.value_coef * critic_loss + entropy_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return total_loss.item()
