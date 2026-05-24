import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

class PPOAgent:
    """
    Proximal Policy Optimization (PPO) Continuous Agent for RLSFLoc.
    Natively models continuous 3D actions (fusion weights parameters) using a Gaussian policy,
    and performs clipped surrogate policy updates.
    """
    def __init__(self, state_dim=9, action_dim=3, lr=0.002, gamma=0.99, eps_clip=0.2, K_epochs=5, entropy_coef=0.01, value_coef=0.5):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        
        # Actor Network: outputs means (mu) for Gaussian distribution
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, action_dim)
        )
        
        # Log standard deviation: parameterized separately for continuous policy exploration
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        
        # Critic Network: estimates state value V(s)
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
        Select continuous 3D action using Gaussian distribution.
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

    def evaluate_action(self, state_t, action_t):
        """
        Evaluate log probabilities, entropy, and state values for updates.
        """
        mu = self.actor(state_t)
        std = torch.exp(self.log_std)
        dist = Normal(mu, std)
        
        log_prob = dist.log_prob(action_t).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        value = self.critic(state_t).squeeze(-1)
        
        return log_prob, value, entropy

    def update(self, states, actions, old_log_probs, returns):
        """
        Perform continuous PPO clipped surrogate policy and value function updates.
        """
        states_t = torch.FloatTensor(np.array(states))
        actions_t = torch.FloatTensor(np.array(actions))
        old_log_probs_t = torch.FloatTensor(np.array(old_log_probs))
        returns_t = torch.FloatTensor(np.array(returns))
        
        losses = []
        for _ in range(self.K_epochs):
            # Evaluate new policies and values
            log_probs, values, entropy = self.evaluate_action(states_t, actions_t)
            
            # Advantages
            advantages_t = returns_t - values.detach()
            # Normalize advantages for training stability
            advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
            
            # Clipped policy ratio
            ratios = torch.exp(log_probs - old_log_probs_t)
            
            surr1 = ratios * advantages_t
            surr2 = torch.clamp(ratios, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantages_t
            
            # Actor, Critic, and Entropy losses
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = self.loss_fn(values, returns_t)
            entropy_loss = -self.entropy_coef * entropy.mean()
            
            total_loss = actor_loss + self.value_coef * critic_loss + entropy_loss
            
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
            
            losses.append(total_loss.item())
            
        return np.mean(losses)
